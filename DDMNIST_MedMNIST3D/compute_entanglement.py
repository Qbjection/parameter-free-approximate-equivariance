import argparse
import torch
from models.GxGRegularFunctorModel import GxGRegularFunctor
from datasets.C4xC4DDMNIST_dataset import C4xC4DDMNISTDataModule
from datasets.D4xD4DDMNIST_dataset import D4xD4DDMNISTDataModule
from datasets.D1xD1DDMNIST_dataset import D1xD1DDMNISTDataModule


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
            outputs, latents = model.model(x1)
            # latents is a list of tensors (one per tracked layer); concatenate them
            latent = torch.cat(latents, dim=-1)

            all_latents.append(latent.cpu())
            all_labels.append(y1)

    all_latents = torch.cat(all_latents, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_latents, all_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str, help='Path to the checkpoint file of the trained model.')
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