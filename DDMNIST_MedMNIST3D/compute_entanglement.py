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

            # Forward pass through the underlying DDMNIST CNN
            outputs, latents = model.model(x1, [12])
            # latents is a list of tensors (one per tracked layer); concatenate them
            latent = torch.cat(latents, dim=-1)

            all_latents.append(latent.cpu())
            all_labels.append(y1)

    all_latents = torch.cat(all_latents, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_latents, all_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint file of the trained model, up to the version directory.')
    parser.add_argument('--versions', type=int, nargs='+', default=[0], help='List of version numbers to compute entanglement for (e.g., 0 1 2).')
    parser.add_argument('--dataset', type=str, default='ddmnist_c4', choices=DATASET_TO_DATAMODULE.keys(),
                        help='Dataset to use for extracting latents.')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Which data split to extract latents from.')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ents = []
    val_ents = []
    test_ents = []

    num_versions = len(args.versions)
    print(f"Computing entanglement for {num_versions} versions: {args.versions}")

    for version in args.versions:
        # Load model from checkpoint
        model = GxGRegularFunctor.load_from_checkpoint(
            args.checkpoint_path + f"version_{version}/checkpoints/best_model.ckpt", 
            map_location=device
        )
        # Ensure the inner CNN returns latents
        model.model.get_latent = True
        model = model.to(device)

        # Set up data
        DataModuleClass = DATASET_TO_DATAMODULE[args.dataset]
        data_module = DataModuleClass(args.batch_size)
        data_module.setup()

        for split in ['train', 'val', 'test']:
            if split == 'train':
                dataloader = data_module.train_dataloader()
            elif split == 'val':
                dataloader = data_module.val_dataloader()
            else:
                dataloader = data_module.test_dataloader()

            # Extract latents
            latents, labels = extract_latents(model, dataloader, device)

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
            else:
                raise NotImplementedError("Entanglement computation is currently only implemented for the DDMNIST C4xC4 dataset.")
            
            if split == "train":
                train_ents.append(avg_entanglement)
            elif split == "val":
                val_ents.append(avg_entanglement)
            elif split == "test":
                test_ents.append(avg_entanglement)

    print(f"Average entanglement across versions:")
    print(f"Train: {sum(train_ents) / len(train_ents):.4f}")
    print(f"Val: {sum(val_ents) / len(val_ents):.4f}")
    print(f"Test: {sum(test_ents) / len(test_ents):.4f}")

    avg_entanglement_random_vectors = get_normalized_average_entanglement(num_samples=1000, dim_a=tensor_latents.shape[1] // rep_dims, dim_b=rep_dims)
    avg_entropy = avg_entanglement_random_vectors.get("normalized_avg_entropy_A")
    print(f"Average normalized entanglement of random vectors: {avg_entropy:.4f}")