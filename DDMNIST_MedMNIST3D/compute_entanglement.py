import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.GxGRegularFunctorModel import GxGRegularFunctor
from datasets.C4xC4DDMNIST_dataset import C4xC4DDMNISTDataModule
from datasets.D4xD4DDMNIST_dataset import D4xD4DDMNISTDataModule
from datasets.D1xD1DDMNIST_dataset import D1xD1DDMNISTDataModule

from utils.entanglement import Entanglement, get_average_entanglement, get_normalized_average_entanglement


def mean_std_str(values):
    arr = np.array(values)
    return f"{arr.mean():.4f} ± {arr.std():.4f}"

DATASET_TO_DATAMODULE = {
    'ddmnist_c4': C4xC4DDMNISTDataModule,
    'ddmnist_d4': D4xD4DDMNISTDataModule,
    'ddmnist_d1': D1xD1DDMNISTDataModule,
}


def extract_latents_and_predictions(model, dataloader, device):
    """Run a forward pass over the dataloader and collect latents, labels, and predicted classes."""
    model.eval()
    all_latents = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            (x1, y1), (x2, y2), transformation_type, covariate = batch
            x1 = x1.to(device)

            # Forward pass through the underlying DDMNIST CNN
            logits, latents = model.model(x1, [12])
            # latents is a list of tensors (one per tracked layer); concatenate them
            latent = torch.cat(latents, dim=-1)
            predictions = torch.argmax(logits, dim=1)

            all_latents.append(latent.cpu())
            all_labels.append(y1)
            all_predictions.append(predictions.cpu())

    all_latents = torch.cat(all_latents, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    return all_latents, all_labels, all_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint file of the trained model, up to the version directory. End this with a slash.')
    parser.add_argument('--versions', type=int, nargs='+', default=[0], help='List of version numbers to compute entanglement for (e.g., 0 1 2).')
    parser.add_argument('--dataset', type=str, default='ddmnist_c4', choices=DATASET_TO_DATAMODULE.keys(),
                        help='Dataset to use for extracting latents.')
    parser.add_argument('--batch_size', type=int, default=256)

    args = parser.parse_args()

    # Seed all RNGs before any dataset/model construction so that (i) the MNIST
    # pairings sampled in DDMNIST._create_data are identical across invocations,
    # and (ii) the rotations sampled in PairedC4xC4DDMNIST.augment_image are
    # identical across invocations.
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build the datasets once so every version evaluates on the same splits.
    DataModuleClass = DATASET_TO_DATAMODULE[args.dataset]
    dm_plain = DataModuleClass(args.batch_size, augment_test=False)
    dm_plain.setup()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    dm_aug = DataModuleClass(args.batch_size, augment_test=True)
    dm_aug.setup()

    # Rebuild the DataLoaders with num_workers=0 so every augmentation call
    # runs in the main process under the seeds set above, instead of in
    # independently-seeded worker subprocesses.
    dataloaders = {
        'test':     DataLoader(dm_plain.test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0),
        'aug_test': DataLoader(dm_aug.test_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0),
    }

    test_ents = []
    aug_test_ents = []

    test_accuracies = []
    aug_test_accuracies = []

    num_versions = len(args.versions)
    print(f"Computing entanglement for {num_versions} versions: {args.versions}")

    for version in args.versions:
        # Re-seed before each version so every version iterates the dataloaders
        # under identical augmentation RNG state (the dataset splits are already
        # fixed, but augment_image consumes random draws as batches are produced).
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # Load model from checkpoint
        model = GxGRegularFunctor.load_from_checkpoint(
            args.checkpoint_path + f"version_{version}/checkpoints/best_model.ckpt",
            map_location=device
        )
        # Ensure the inner CNN returns latents
        model.model.get_latent = True
        model = model.to(device)

        for split in ['test', 'aug_test']:
            dataloader = dataloaders[split]

            # Extract latents and predictions
            latents, labels, predictions = extract_latents_and_predictions(model, dataloader, device)
            split_accuracy = (predictions == labels).float().mean().item()

            if args.dataset == 'ddmnist_c4':
                # For C4xC4, the regular representation is 16-dim
                rep_dims = 16
                tensor_dims = int((latents.shape[1] // rep_dims) * rep_dims)
            elif args.dataset == 'ddmnist_d1':
                rep_dims = 4
                tensor_dims = 64 #TODO THIS IS HARDCODED
            elif args.dataset == 'ddmnist_d4':
                rep_dims = 8
                tensor_dims = int((latents.shape[1] // rep_dims) * rep_dims)
            else:
                raise NotImplementedError("Entanglement computation is currently only implemented for the DDMNIST C4xC4 dataset.")
            
            tensor_latents = latents[:, :tensor_dims]
            # extracted part of the latent space that corresponds to operations with the regular representation
            print(f"Tensor latents shape (should be divisible by {rep_dims}): {tensor_latents.shape}")
            norm_tensor_latents = tensor_latents / torch.linalg.vector_norm(tensor_latents, dim=1, keepdim=True)
            ent = Entanglement(norm_tensor_latents, tensor_latents.shape[1] // rep_dims, rep_dims)
            avg_entanglement = ent.compute(normalize=True).get("entanglement_a").mean().item()
            
            if split == "test":
                test_ents.append(avg_entanglement)
                test_accuracies.append(split_accuracy)
            elif split == "aug_test":
                aug_test_ents.append(avg_entanglement)
                aug_test_accuracies.append(split_accuracy)

    print(f"Entanglement across versions (mean +/- std):")
    print(f"Test:     {mean_std_str(test_ents)}")
    print(f"Aug Test: {mean_std_str(aug_test_ents)}")

    print(f"\nAccuracy across versions (mean +/- std):")
    print(f"Test:     {mean_std_str(test_accuracies)}")
    print(f"Aug Test: {mean_std_str(aug_test_accuracies)}")

    avg_entanglement_random_vectors = get_normalized_average_entanglement(num_samples=1000, dim_a=tensor_latents.shape[1] // rep_dims, dim_b=rep_dims)
    avg_entropy = avg_entanglement_random_vectors.get("normalized_avg_entropy_A")
    print(f"Average normalized entanglement of random vectors: {avg_entropy:.4f}")