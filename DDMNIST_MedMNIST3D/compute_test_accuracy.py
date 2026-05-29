import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.GxGRegularFunctorModel import GxGRegularFunctor
from datasets.C4xC4DDMNIST_dataset import C4xC4DDMNISTDataModule
from datasets.D4xD4DDMNIST_dataset import D4xD4DDMNISTDataModule
from datasets.D1xD1DDMNIST_dataset import D1xD1DDMNISTDataModule

DATASET_TO_DATAMODULE = {
    'ddmnist_c4': C4xC4DDMNISTDataModule,
    'ddmnist_d4': D4xD4DDMNISTDataModule,
    'ddmnist_d1': D1xD1DDMNISTDataModule,
}
BATCH_SIZE = 256


def seed_all():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


@torch.no_grad()
def test_accuracy(model, dataloader, device):
    model.eval()
    correct = total = 0
    for (x1, y1), *_ in dataloader:
        out = model(x1.to(device))
        logits = out[0] if isinstance(out, tuple) else out
        pred = torch.argmax(logits, dim=1).cpu()
        correct += (pred == y1).sum().item()
        total += y1.numel()
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_paths', type=str, nargs='+', required=True,
                        help='Full paths to one or more trained model checkpoint (.ckpt) files.')
    parser.add_argument('--dataset', type=str, default='ddmnist_c4', choices=DATASET_TO_DATAMODULE.keys(),
                        help='Dataset to use for extracting test accuracy.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # plain and augmented test loaders
    DataModuleClass = DATASET_TO_DATAMODULE[args.dataset]
    seed_all()
    dm_plain = DataModuleClass(BATCH_SIZE, augment_test=False)
    dm_plain.setup()
    dm_aug = DataModuleClass(BATCH_SIZE, augment_test=True)
    dm_aug.setup()
    plain_loader = DataLoader(dm_plain.test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    aug_loader = DataLoader(dm_aug.test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    header = f"{'lambda_t':>10} {'lambda_e':>10} {'test_acc':>10} {'aug_test_acc':>14}"
    print(header)
    print("-" * len(header))
    for path in args.checkpoint_paths:
        seed_all()
        model = GxGRegularFunctor.load_from_checkpoint(path, map_location=device).to(device)
        plain_acc = test_accuracy(model, plain_loader, device)
        seed_all()
        aug_acc = test_accuracy(model, aug_loader, device)
        lt = model.hparams['lambda_t']
        le = model.hparams['lambda_e']
        print(f"{lt:>10} {le:>10} {plain_acc:>10.4f} {aug_acc:>14.4f}")
