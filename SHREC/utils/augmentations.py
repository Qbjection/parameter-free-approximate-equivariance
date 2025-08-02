"""
Augmentation functions for spherical signals.

This module provides efficient augmentation functions for 2D spherical data,
particularly for the SHREC dataset. The augmentations preserve the spherical
geometry while applying various transformations.
"""

import jax
import jax.numpy as jnp
from typing import List, Union


@jax.jit
def d4_shrec_augment(x: jnp.ndarray, transform_indices: Union[List[int], jnp.ndarray]) -> jnp.ndarray:
    """
    Apply D4 (dihedral group) transformations to spherical signals.
    
    The D4 group consists of 8 transformations:
    - 4 rotations: 0°, 90°, 180°, 270° (about z-axis/longitude)
    - 4 reflections: reflection about latitude=π/2 followed by each rotation
    
    Args:
        x: Input spherical signals with shape [B, H, W, C]
           where H is latitude dimension and W is longitude dimension
        transform_indices: List or array of integers of length B, each in range [0, 7]
                          indexing which D4 transformation to apply to each batch element
    
    Returns:
        Transformed signals with same shape [B, H, W, C]
        
    Transformations:
        0: Identity (no transformation)
        1: 90° rotation about z-axis
        2: 180° rotation about z-axis  
        3: 270° rotation about z-axis
        4: Reflection about equator (latitude flip)
        5: Reflection + 90° rotation
        6: Reflection + 180° rotation
        7: Reflection + 270° rotation
    """
    B, H, W, C = x.shape
    transform_indices = jnp.asarray(transform_indices)
    
    # Note: Assertions removed for JIT compatibility
    # JAX JIT cannot compile assertion statements
    
    def apply_single_transform(xi: jnp.ndarray, transform_idx: int) -> jnp.ndarray:
        """Apply a single D4 transformation to one sample."""
        # xi has shape [H, W, C]
        
        # Extract rotation and reflection components
        rotation_idx = transform_idx % 4  # 0, 1, 2, 3 for 0°, 90°, 180°, 270°
        apply_reflection = transform_idx >= 4
        
        # Apply reflection first (flip latitude dimension) - use jnp.where for JAX compatibility
        xi = jnp.where(apply_reflection, jnp.flip(xi, axis=0), xi)
        
        # Apply rotation (shift along longitude dimension)
        # For spherical coordinates, 90° = W/4, 180° = W/2, 270° = 3*W/4
        shift_90 = W // 4
        shift_180 = W // 2  
        shift_270 = (3 * W) // 4
        
        # Create all possible rotations
        xi_rot0 = xi  # No rotation
        xi_rot1 = jnp.roll(xi, shift=shift_90, axis=1)    # 90° rotation
        xi_rot2 = jnp.roll(xi, shift=shift_180, axis=1)   # 180° rotation  
        xi_rot3 = jnp.roll(xi, shift=shift_270, axis=1)   # 270° rotation
        
        # Select the appropriate rotation using jnp.select
        xi = jnp.select(
            [rotation_idx == 0, rotation_idx == 1, rotation_idx == 2, rotation_idx == 3],
            [xi_rot0, xi_rot1, xi_rot2, xi_rot3],
            default=xi_rot0
        )
            
        return xi
    
    # Vectorized application using vmap
    return jax.vmap(apply_single_transform)(x, transform_indices)


def random_d4_indices(key: jax.random.PRNGKey, batch_size: int) -> jnp.ndarray:
    """
    Generate random D4 transformation indices for a batch.
    
    Args:
        key: JAX random key
        batch_size: Number of indices to generate
        
    Returns:
        Array of shape [batch_size] with random integers in [0, 7]
    """
    return jax.random.randint(key, shape=(batch_size,), minval=0, maxval=8)


def d4_transformation_names() -> List[str]:
    """
    Get human-readable names for D4 transformations.
    
    Returns:
        List of 8 transformation names corresponding to indices 0-7
    """
    return [
        "identity",
        "rot_90",
        "rot_180", 
        "rot_270",
        "reflect",
        "reflect_rot_90",
        "reflect_rot_180",
        "reflect_rot_270"
    ]


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    # Create dummy spherical data
    B, H, W, C = 4, 32, 64, 16
    x = jnp.ones((B, H, W, C))
    
    # Add some pattern to visualize transformations
    for i in range(B):
        for j in range(H):
            for k in range(W):
                x = x.at[i, j, k, 0].set(jnp.sin(j * jnp.pi / H) * jnp.cos(k * 2 * jnp.pi / W))
    
    # Test with different transformation indices
    transform_indices = [0, 1, 4, 7]  # identity, 90° rot, reflect, reflect+270° rot
    
    x_transformed = d4_shrec_augment(x, transform_indices)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_transformed.shape}")
    print(f"Transform indices: {transform_indices}")
    print(f"Transform names: {[d4_transformation_names()[i] for i in transform_indices]}")
    
    # Verify transformations preserve overall statistics
    print(f"Input mean: {jnp.mean(x):.6f}")
    print(f"Output mean: {jnp.mean(x_transformed):.6f}")
    print(f"Input std: {jnp.std(x):.6f}")
    print(f"Output std: {jnp.std(x_transformed):.6f}")
