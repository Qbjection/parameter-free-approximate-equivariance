import argparse
import torch
from models.GxGRegularFunctorModel import GxGRegularFunctor
from datasets.C4xC4DDMNIST_dataset import C4xC4DDMNISTDataModule
from datasets.D4xD4DDMNIST_dataset import D4xD4DDMNISTDataModule
from datasets.D1xD1DDMNIST_dataset import D1xD1DDMNISTDataModule

from utils.entanglement import Entanglement, get_average_entanglement, get_normalized_average_entanglement

DATASET_TO_DATAMODULE = {
    'ddmnist_c4': C4xC4DDMNISTDataModule,
    'ddmnist_d4': D4xD4DDMNISTDataModule,
    'ddmnist_d1': D1xD1DDMNISTDataModule,
}


def extract_latents(model, dataloader, device):
    """Run a forward pass over the dataloader and collect latent vectors."""
    model.eval()
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            (x1, y1), (x2, y2), transformation_type, covariate = batch
            x1 = x1.to(device)
            print("x1 shape:", x1.shape)  # Debug print to check input shape

            # Forward pass through the underlying DDMNIST CNN
            outputs, latents = model.model(x1, [12])
            print("Outputs shape:", outputs.shape)  # Debug print to check output shape
            print("Number of latent tensors:", len(latents))  # Debug print to check number of latent tensors
            # latents is a list of tensors (one per tracked layer); concatenate them
            latent = torch.cat(latents, dim=-1)

            all_latents.append(latent.cpu())
            all_labels.append(y1)

    all_latents = torch.cat(all_latents, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_latents, all_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint file of the trained model.')
    parser.add_argument('--dataset', type=str, default='ddmnist_c4', choices=DATASET_TO_DATAMODULE.keys(),
                        help='Dataset to use for extracting latents.')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Which data split to extract latents from.')

    args = parser.parse_args()

    # Load model from checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GxGRegularFunctor.load_from_checkpoint(args.checkpoint_path, map_location=device)
    # Ensure the inner CNN returns latents
    model.model.get_latent = True
    model = model.to(device)

    # Set up data
    DataModuleClass = DATASET_TO_DATAMODULE[args.dataset]
    data_module = DataModuleClass(args.batch_size)
    data_module.setup()

    if args.split == 'train':
        dataloader = data_module.train_dataloader()
    elif args.split == 'val':
        dataloader = data_module.val_dataloader()
    else:
        dataloader = data_module.test_dataloader()

    # Extract latents
    latents, labels = extract_latents(model, dataloader, device)
    print(f"Extracted latents shape: {latents.shape}")
    print(f"Labels shape: {labels.shape}")

    if args.dataset == 'ddmnist_c4':
        # For C4xC4, the regular representation is 16-dim
        rep_dims = 16
        tensor_dims = int((latents.shape[1] // rep_dims) * rep_dims)
        tensor_latents = latents[:, :tensor_dims]

        # extracted part of the latent space that corresponds to operations with the regular representation
        print(f"Tensor latents shape (should be divisible by {rep_dims}): {tensor_latents.shape}")

        norm_tensor_latents = tensor_latents / torch.linalg.vector_norm(tensor_latents, dim=1, keepdim=True)


        ent = Entanglement(norm_tensor_latents, tensor_latents.shape[1] // rep_dims, rep_dims)
        avg_entanglement = ent.compute(normalize=True).get("entanglement_a").mean().item()

        print(f"Average normalized entanglement of {args.split} vectors: {avg_entanglement:.4f}")

        avg_entanglement_random_vectors = get_normalized_average_entanglement(num_samples=1000, dim_a=tensor_latents.shape[1] // rep_dims, dim_b=rep_dims)
        print(f"Average normalized entanglement of random vectors: {avg_entanglement_random_vectors:.4f}")