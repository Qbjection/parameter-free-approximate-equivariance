
"""
Train.

Usage:
  train [options] [--linear] [--projection=<type>] [--freeze-encoder]
  train (-h | --help)
  train --version

Options:
  --linear                Use linear latent transformation (matrix W) instead of kernel (default: False).
  --projection=<type>     Projection type for linear mode: matrix, mlp, or conv [default: matrix].
  --freeze-encoder        Freeze encoder weights during training (default: True).
  -h --help                   Show this screen.
  --version                   Show version.
  -i, --in=<input_dir>        Input directory [default: {root_path}/data/CSHREC_11/].
  -w, --weights=<weight_dir>  Directory with pre-trained model weights from cshrec11_encode.
  -o, --out=<output_dir>      Output directory [default: {root_path}/experiments/cshrec11_pred/weights/].
  -c, --config=<config>       Config directory [default: config].

"""

import functools
import os
import sys
import importlib.util
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds 
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.cm as cm 
import wandb
import scipy as sp 
import io

print(jax.default_backend(), flush=True)
tf.config.experimental.set_visible_devices( [], 'GPU' )

from flax.core.frozen_dict import freeze 
from docopt import docopt
from typing import Any, Callable, Dict, Sequence, Tuple, Union
from clu import checkpoint, metric_writers, metrics, parameter_overview, periodic_actions
from tqdm import tqdm
from icecream import ic

from os.path import dirname, abspath
ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.append( ROOT_PATH )

from nn import losses, conv, fmaps, equiv
from utils import utils
from data import data_loader as dl 
from data import input_pipeline, xforms


# Constants
PMAP_AXIS = "batch" 

######################

# Projection Classes for Linear Mode
class MLPProjection(nn.Module):
    hidden_dim: int
    output_dim: int
    
    @nn.compact
    def __call__(self, x):
        # x shape: (batch, spatial_dim, latent_dim)
        # Two dense layers with same dimensions as requested
        x = nn.Dense(self.output_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

class ConvProjection(nn.Module):
    target_spatial_dim: int
    latent_dim: int
    encoder_spatial_h: int  # zH
    encoder_spatial_w: int  # zW
    
    @nn.compact
    def __call__(self, x):
        # x shape: (batch, spatial_dim, latent_dim)
        batch_size, spatial_dim, latent_dim = x.shape
        
        # Use the actual encoder spatial dimensions
        zH, zW = self.encoder_spatial_h, self.encoder_spatial_w
        
        # Verify our understanding matches
        assert zH * zW == spatial_dim, f"Spatial dimension mismatch: {zH}*{zW}={zH*zW} != {spatial_dim}"
        
        # Reshape to actual encoder output spatial layout
        x = jnp.reshape(x, (batch_size, zH, zW, latent_dim))
        
        # Convolutional layers to reduce spatial dimensions
        x = nn.Conv(features=latent_dim, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        
        # Calculate target spatial dimensions (assuming square for target)
        target_h = target_w = int(jnp.sqrt(self.target_spatial_dim))
        
        # Adaptive downsampling
        if zH > target_h or zW > target_w:
            stride_h = max(1, zH // target_h)
            stride_w = max(1, zW // target_w)
            stride = max(stride_h, stride_w)  # Use larger stride to ensure we don't exceed target
            
            x = nn.Conv(features=latent_dim, kernel_size=(3, 3), strides=(stride, stride), padding='SAME')(x)
            x = nn.relu(x)
        
        # Final convolution
        x = nn.Conv(features=latent_dim, kernel_size=(1, 1))(x)
        
        # Get actual output spatial dimensions after convolutions
        actual_h, actual_w = x.shape[1], x.shape[2]
        actual_spatial_dim = actual_h * actual_w
        
        # Reshape back to flattened spatial
        x = jnp.reshape(x, (batch_size, actual_spatial_dim, latent_dim))
        
        # Final dense layer to get exact target dimension if needed
        if actual_spatial_dim != self.target_spatial_dim:
            x = nn.Dense(self.target_spatial_dim)(x.transpose(0, 2, 1)).transpose(0, 2, 1)
        
        return x

class GeometricProjection(nn.Module):
    target_spatial_dim: int
    latent_dim: int
    projection_type: str = "matrix"  # "matrix", "mlp", or "conv"
    
    @nn.compact  
    def __call__(self, x):
        batch_size, spatial_dim, latent_dim = x.shape
        
        if self.projection_type == "conv" and spatial_dim >= 16:  # Use conv for reasonable spatial dims
            # Reshape to 2D spatial for convolution
            spatial_size = int(jnp.sqrt(spatial_dim))
            x = jnp.reshape(x, (batch_size, spatial_size, spatial_size, latent_dim))
            
            # Convolutional reduction
            x = nn.Conv(features=latent_dim, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.relu(x)
            
            # Adaptive pooling to target size
            target_size = int(jnp.sqrt(self.target_spatial_dim))
            if spatial_size > target_size:
                stride = max(1, spatial_size // target_size)
                x = nn.Conv(features=latent_dim, kernel_size=(3, 3), strides=(stride, stride), padding='SAME')(x)
                x = nn.relu(x)
            
            x = jnp.reshape(x, (batch_size, -1, latent_dim))
            
            # Final adjustment to exact target dimension if needed
            if x.shape[1] != self.target_spatial_dim:
                x = nn.Dense(self.target_spatial_dim)(x.transpose(0, 2, 1)).transpose(0, 2, 1)
                
        elif self.projection_type == "mlp":
            # Use MLP projection with 2 dense layers
            # First layer: project latent features 
            x = nn.Dense(self.target_spatial_dim)(x)
            x = nn.silu(x)
            
            # Now project spatial dimension from enc_spatial_dim to target_spatial_dim
            x = nn.Dense(self.target_spatial_dim)(x.transpose(0, 2, 1)).transpose(0, 2, 1)
            
        else:  # "matrix" - simple linear projection
            # Simple linear projection via einsum (as before)
            # This will be handled in the loss_fn to maintain backward compatibility
            return x
        
        return x

@flax.struct.dataclass
class TrainState:
  step         : int
  opt_state    : Any
  params       : Any
  key          : Any 


def create_train_state( cfg: Any, data_shape: Tuple, num_classes: int, linear: bool = False, proj_type: str = "matrix") -> Tuple[nn.Module, Any, Any, Any,  Any, TrainState, Any]:

  # Random key 
  seed = 0 #np.random.randint(low=0, high=1e8, size=(1, ))[0]

  key                              = jax.random.PRNGKey( seed )
  key, enc_key, model_key, kernel_key, xform_key = jax.random.split( key, 5 )
  
  down_factor = np.power( 2, (len(cfg.CONV_ENC_CHANNELS) - 1) )

  ae_state = checkpoint.load_state_dict(cfg.PRE_TRAIN_DIR) 
  
  enc_params = ae_state["params"]["encoder"]
  kernel_params = ae_state["params"]["kernel"]

  # Models
  encoder = conv.ConvEncoder( channels    = cfg.CONV_ENC_CHANNELS,
                              block_depth = cfg.CONV_ENC_BLOCK_DEPTH,
                              kernel_size = cfg.KERNEL_SIZE,
                              out_dim     = cfg.LATENT_DIM)

  dec_input    = jnp.ones( (data_shape[0], data_shape[1] // down_factor, data_shape[2] // down_factor, cfg.LATENT_DIM), dtype=jnp.float32 )
                     
  kernel_input = jnp.reshape(dec_input, (dec_input.shape[0], -1,  cfg.LATENT_DIM))
            
  kernel = fmaps.operator_iso(op_dim=cfg.KERNEL_OP_DIM)
  
  # Initialize
  enc_input     = jnp.ones( data_shape, dtype=jnp.float32 )

  _             = encoder.init(enc_key, enc_input)
  _             = kernel.init(enc_key, kernel_input, kernel_input)

  
  model = equiv.orth_net(  features      = cfg.MLP_FEATURES,
                    num_layers    = cfg.MLP_LAYERS,
                    out_dim       = num_classes)
  
  
  # Calculate encoder output spatial dimension
  down_factor = np.power(2, (len(cfg.CONV_ENC_CHANNELS) - 1))
  enc_spatial_dim = (data_shape[1] // down_factor) * (data_shape[2] // down_factor)
  
  if linear:
    # In linear mode, use encoder output dimensions and add projection
    xform_dim = cfg.KERNEL_OP_DIM  # Target dimension after projection
    model_input = jnp.ones( (data_shape[0], xform_dim, cfg.LATENT_DIM), dtype=jnp.float32)
    
    # Create projection based on type
    if proj_type == "matrix":
      # Simple matrix projection (backward compatibility)
      proj_key, model_key = jax.random.split(model_key)
      projection_matrix = jax.random.normal(proj_key, (enc_spatial_dim, cfg.KERNEL_OP_DIM)) * 0.01
      projection_obj = None
    else:
      # Calculate actual encoder spatial dimensions (zH, zW)
      encoder_h = data_shape[1] // down_factor  # This is zH
      encoder_w = data_shape[2] // down_factor  # This is zW
      
      # Verify our calculation
      assert encoder_h * encoder_w == enc_spatial_dim, f"Spatial calc mismatch: {encoder_h}*{encoder_w}={encoder_h*encoder_w} != {enc_spatial_dim}"
      
      if proj_type == "conv":
        projection_obj = ConvProjection(
          target_spatial_dim=cfg.KERNEL_OP_DIM,
          latent_dim=cfg.LATENT_DIM,
          encoder_spatial_h=encoder_h,  # Pass zH
          encoder_spatial_w=encoder_w   # Pass zW
        )
      else:  # mlp
        projection_obj = MLPProjection(
          hidden_dim=cfg.KERNEL_OP_DIM,  # Keep for compatibility but not used
          output_dim=cfg.KERNEL_OP_DIM
        )
      
      # Initialize projection
      proj_input = jnp.ones((data_shape[0], enc_spatial_dim, cfg.LATENT_DIM), dtype=jnp.float32)
      proj_key, model_key = jax.random.split(model_key)
      projection_params = projection_obj.init(proj_key, proj_input)["params"]
      projection_matrix = None
  else:
    # In kernel mode, use kernel operator dimensions
    xform_dim = cfg.KERNEL_OP_DIM
    model_input = jnp.ones( (data_shape[0], xform_dim, cfg.LATENT_DIM), dtype=jnp.float32)
    projection_matrix = None
    projection_obj = None

  model_params = model.init(model_key, model_input)["params"]

  if linear:
    if proj_type == "matrix":
      params = {"encoder": enc_params, "kernel": kernel_params, "model": model_params, "projection": projection_matrix}
    else:
      params = {"encoder": enc_params, "kernel": kernel_params, "model": model_params, "projection_obj": projection_params}
  else:
    params = {"encoder": enc_params, "kernel": kernel_params, "model": model_params} 


  # Set up optimizer 
  schedule = optax.warmup_cosine_decay_schedule( init_value   = cfg.INIT_LR,
                                                 peak_value   = cfg.LR,
                                                 warmup_steps = cfg.WARMUP_STEPS,
                                                 decay_steps  = cfg.NUM_TRAIN_STEPS,
                                                 end_value    = cfg.END_LR )

  batch_size = cfg.BATCH_SIZE
  num_multi_steps = cfg.TRUE_BATCH_SIZE // batch_size  
  
  optim = optax.adamw( learning_rate=schedule, b1=cfg.ADAM_B1, b2=cfg.ADAM_B2 )
  
  optimizer = optax.MultiSteps( optim, num_multi_steps )
  
  
  state       = optimizer.init( params ) 
  train_state = TrainState( step=0, opt_state=state, params=params, key=key )

  return model, encoder, kernel, optimizer, xform_key, train_state, projection_obj

@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  train_loss: metrics.Average.from_output("train_loss") 
  train_acc: metrics.Average.from_output("train_acc")

@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  eval_loss: metrics.Average.from_output("eval_loss")
  eval_acc: metrics.Average.from_output("eval_acc")
  

def train_step(x: Any, labels: Any, model: nn.Module, encoder: nn.Module, kernel: Any, 
               state: TrainState, optimizer: Any, train: bool = True, linear: bool = False, projection_obj: Any = None, freeze_encoder: bool = True):
  
  key  = state.key 
  step = state.step + 1
  
  def loss_fn(params):
    z = encoder.apply({"params": params["encoder"]}, x)
    z = jnp.reshape(z, (z.shape[0], -1, z.shape[-1]))
    
    # Conditionally freeze encoder based on command line argument
    if freeze_encoder:
      z = jax.lax.stop_gradient(z)

    if linear:
      # Apply learnable projection from enc_spatial_dim to KERNEL_OP_DIM
      # z shape: (batch, enc_spatial_dim, latent_dim)
      if "projection" in params:
        # Matrix projection (backward compatibility)
        # projection_matrix shape: (enc_spatial_dim, KERNEL_OP_DIM)
        # Result: (batch, KERNEL_OP_DIM, latent_dim)
        z_proj = jnp.einsum('bsl,sk->bkl', z, params["projection"])
      else:
        # Use GeometricProjection (mlp/conv)
        z_proj = projection_obj.apply({'params': params["projection_obj"]}, z)
      
      logits = model.apply({'params': params["model"]}, z_proj)
      
      # Return dummy Omega for downstream compatibility
      batch_size = z.shape[0]
      num_latents = z.shape[1]
      Omega = (
          jnp.zeros((num_latents, num_latents), dtype=jnp.float32),   # Omega[0]
          jnp.zeros((num_latents,), dtype=jnp.float32),               # Omega[1]
          jnp.zeros((num_latents,), dtype=jnp.float32)                # Omega[2]
      )
    else:
      # KERNEL MODE: original logic
      _, Omega = kernel.apply({"params": params["kernel"]}, z, z)
      z0_b = jnp.einsum("...lj, ...lm->...jm", Omega[0][None, ...], Omega[2][None, ..., None] * z)
      logits = model.apply({'params': params["model"]}, z0_b)

    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, (logits,) 
 
  if train: 

    # Compute gradient
    grad_fn = jax.value_and_grad( loss_fn, has_aux=True )

    (loss, (logits, )), grad = grad_fn(state.params)

    grad = jax.lax.pmean( grad, axis_name=PMAP_AXIS )

    grad = jax.tree.map( jnp.conj, grad )

    updates, opt_state = optimizer.update( grad, state.opt_state, state.params )
    params             = optax.apply_updates( state.params, updates )

    new_state = state.replace( step      = step,
                               opt_state = opt_state,
                               params    = params,
                               key       = key )
    
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)
    
    updater = TrainMetrics.gather_from_model_output
      
    metrics_update = updater( train_loss        = loss, 
                              train_acc         = acc)
  else:
    loss, (logits, ) = loss_fn( state.params )

    acc = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics_update = EvalMetrics.single_from_model_output( eval_loss       = loss, 
                                                           eval_acc        = acc)
    
                                                         
    new_state = state.replace( key=key )
    
  return new_state, metrics_update
  


       
def train_and_evaluate( cfg: Any, input_dir: str, output_dir: str, load_weight_dir: str, linear: bool = False, proj_type: str = "matrix", freeze_encoder: bool = True):
  tf.io.gfile.makedirs( output_dir )

  '''
  ========== Setup W&B =============
  '''
  
  # Extract experiment number from weight directory
  # weight_dir format: .../experiments/cshrec11_encode/weights/0/ or similar
  weight_exp_num = "unknown"
  weight_has_linear = False
  
  if load_weight_dir:
    # Check if "linear" is in the path
    weight_has_linear = "linear" in load_weight_dir.lower()
    
    # Extract the experiment number (last directory in the path that's numeric)
    path_parts = load_weight_dir.rstrip('/').split('/')
    for part in reversed(path_parts):
      if part.isdigit():
        weight_exp_num = part
        break
  
  # Create weight suffix with linear indicator if present
  if weight_has_linear:
    weight_suffix = f"linear_weights_exp_{weight_exp_num}"
  else:
    weight_suffix = f"weights_exp_{weight_exp_num}"
  
  # base_name = "niso_{}_kop_{}_ld_{}d_cshrec11_pred".format( cfg.KERNEL_OP_DIM,
  #                                                           cfg.LATENT_DIM,
  #                                                           len(cfg.CONV_ENC_CHANNELS)-1)
  base_name = "cshrec11_pred"
  # Add suffixes based on configuration
  suffixes = []
  
  # Add weights experiment number with latent indicator
  suffixes.append(weight_suffix)
  
  if linear:
    suffixes.append(f"linear_{proj_type}")
  else:
    suffixes.append("kernel")
    
  if not freeze_encoder:
    suffixes.append("unfrozen_enc")
  else:
    suffixes.append("frozen_enc")
  
  exp_name = base_name + "_" + "_".join(suffixes)
  
  project_name      = cfg.PROJECT_NAME 
  cfg.WORK_DIR      = output_dir 
  cfg.PRE_TRAIN_DIR = load_weight_dir 
  cfg.CSHREC11_DIR    = input_dir 
  module_to_dict = lambda module: {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}
  config         = module_to_dict( cfg )

  run = wandb.init(config=config, project=project_name, name=exp_name)
  
  '''
  ==================================
  '''
  

  ## Set up meta-parameters 
  true_batch_size = cfg.TRUE_BATCH_SIZE
  batch_size      = cfg.BATCH_SIZE 
  latent_dim      = cfg.LATENT_DIM 
  
  num_multi_steps  = true_batch_size // batch_size 
    
  eval_fn         = train_step
  train_fn        = train_step

  input_size = cfg.INPUT_SIZE
  eval_every = cfg.EVAL_EVERY

  num_down = len(cfg.CONV_ENC_CHANNELS) - 1  
  latent_size = (input_size[0] // (2**num_down), input_size[1] // (2**num_down))

  num_train_steps = cfg.NUM_TRAIN_STEPS * num_multi_steps
  
  ## Get dataset 
  print( "Getting dataset..." )
  train_data, test_data, NUM_CLASSES, stats = input_pipeline.get_cshrec11(cfg.CSHREC11_DIR)
  
  num_train_repeat = (cfg.NUM_TRAIN_STEPS * batch_size // cfg.DATA_TRAIN_SIZE) + 2
  num_test_repeat  = ((cfg.NUM_TRAIN_STEPS * batch_size * cfg.NUM_EVAL_STEPS * cfg.DATA_TEST_SIZE) // (eval_every * 8)) + 1 
  
  train_data = train_data.map( dl.zbound(stats), num_parallel_calls=tf.data.AUTOTUNE )
  train_data = train_data.shuffle( buffer_size=300, reshuffle_each_iteration=True )
  train_data = train_data.repeat( num_train_repeat )
  train_data = train_data.batch( batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE )

  test_data = test_data.map( dl.zbound(stats), num_parallel_calls=tf.data.AUTOTUNE )
  test_data = test_data.batch( batch_size, drop_remainder=False, num_parallel_calls=tf.data.AUTOTUNE )
  

  train_iter = iter(train_data)
  print( "Done..." )

  #Create models
  print( "Initializing models..." )    
    
  model, encoder, kernel, optimizer, xform_key, state, projection_obj = create_train_state(cfg, (batch_size, *input_size, 16), NUM_CLASSES, linear=linear, proj_type=proj_type)
  
  print( "Done..." )

  # Create checkpoints
  checkpoint_dir = os.path.join( output_dir, "checkpoints" )
  ckpt           = checkpoint.MultihostCheckpoint( checkpoint_dir, max_to_keep=2 )
  state          = ckpt.restore_or_initialize( state )
  
  initial_step = int(state.step) + 1

  
  # Distribute
  state = flax_utils.replicate( state )

  print( "Distributing..." )
  p_train_step = jax.pmap( functools.partial(train_step,
                                             model        = model,
                                             encoder      = encoder,
                                             kernel       = kernel,
                                             optimizer    = optimizer,
                                             train        = True,
                                             linear       = linear,
                                             projection_obj = projection_obj,
                                             freeze_encoder = freeze_encoder),
                           axis_name=PMAP_AXIS )
    
  p_eval_step = jax.pmap( functools.partial(train_step,
                                             model        = model,
                                             encoder      = encoder,
                                             kernel       = kernel,
                                             optimizer    = optimizer,
                                             train        = False,
                                             linear       = linear,
                                             projection_obj = projection_obj,
                                             freeze_encoder = freeze_encoder),
                          axis_name=PMAP_AXIS )
  

  
  # Visualize 
  train_metrics = None

  if cfg.STOP_AFTER is None:
    stop_at = num_train_steps + 1

  else:
    stop_at = cfg.STOP_AFTER + 1 
    
  print( "Beginning training..." )


  for step in tqdm( range(initial_step, stop_at) ):
    is_last_step = step == stop_at - 1

     
    batch = next(train_iter)
    batchIm = jnp.asarray(batch["image"].numpy())
    labels = jnp.asarray(batch["label"].numpy())


    xform_key, batch_key = jax.random.split(xform_key)
    
    batchIm = xforms.draw_shrec11(batchIm, batch_key)

    labels = labels[:, 0]
    

    state, metrics_update = p_train_step(x=batchIm[None, ...], labels=labels[None, ...], state=state)
    metric_update         = flax_utils.unreplicate(metrics_update)
      
    train_metrics = (metric_update if train_metrics is None else train_metrics.merge(metric_update))

      
    '''
    ===========================================
    ============== Eval Loop ==================
    ===========================================
    '''

     
    if step % cfg.EVAL_EVERY == 0 or is_last_step:
      eval_metrics = None 
      
      for batch in tqdm(test_data):
        batchIm = jnp.asarray(batch["image"].numpy())
        labels = jnp.asarray(batch["label"].numpy())

        for j in range(batchIm.shape[1]):
          
          state, emetrics_update = p_eval_step(x=batchIm[None, :, j, ...], labels=labels[None, :, j], state=state)
          emetric_update = flax_utils.unreplicate(emetrics_update)
  
          eval_metrics = (emetric_update if eval_metrics is None else eval_metrics.merge(emetric_update)) 

      run.log(data=eval_metrics.compute(), step=step)

      
    if step % cfg.LOG_LOSS_EVERY == 0 or is_last_step:
      run.log(data=train_metrics.compute(), step=step)
      train_metrics = None
   
    if step % cfg.CHECKPOINT_EVERY == 0 or is_last_step:
      ckpt.save( flax_utils.unreplicate(state) )

  
'''
#######################################################
###################### Main ###########################
#######################################################
'''

if __name__ == '__main__':
  arguments = docopt(__doc__, version='Train 1.0')
  linear = arguments['--linear']
  proj_type = arguments['--projection']
  freeze_encoder = arguments['--freeze-encoder']

  # Set up experiment directory
  in_dir = arguments['--in']
  in_dir = in_dir.format( root_path=ROOT_PATH )

  out_dir = arguments['--out']
  out_dir = out_dir.format( root_path=ROOT_PATH )
  if linear:
    out_dir = f"{out_dir}linear"

  weight_dir = arguments['--weights']

  config = arguments['--config']
  path   = dirname( abspath(__file__) )

  spec                       = importlib.util.spec_from_file_location( "config", f"{path}/configs/{config}.py" )
  cfg                        = importlib.util.module_from_spec(spec)
  sys.modules["module.name"] = cfg
  spec.loader.exec_module( cfg )

  if not os.path.exists( out_dir ):
    os.makedirs( out_dir )

  files = os.listdir( out_dir )
  count = 0
  
  for f in files:

    if os.path.isdir( os.path.join(out_dir, f) ) and f.isnumeric():
      count += 1

  exp_num = str( count )
  
  print( "==========================" )
  print( f"Experiment # {exp_num}" )
  print( "==========================" )
  
  exp_dir = os.path.join( out_dir, exp_num )
  os.mkdir( exp_dir )
  ic( exp_dir )

  
  train_and_evaluate( cfg, in_dir, exp_dir, weight_dir, linear=linear, proj_type=proj_type, freeze_encoder=freeze_encoder)

