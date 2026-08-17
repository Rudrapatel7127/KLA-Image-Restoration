from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class WaferDataset(Dataset):
    def __init__(self, root, augment=True):
        root = Path(root)

        self.augment = augment
        self.gt_dir = root / "GT"
        self.noisy_dir = root / "NoisyLR"

        if not self.gt_dir.is_dir():
            raise FileNotFoundError(
                f"Ground-truth directory not found: {self.gt_dir}. "
                f"Expected the '__init__' root to contain a 'GT' sub-folder."
            )

        if not self.noisy_dir.is_dir():
            raise FileNotFoundError(
                f"Noisy-LR directory not found: {self.noisy_dir}. "
                f"Expected the '__init__' root to contain a 'NoisyLR' sub-folder."
            )

        self.files = sorted(self.gt_dir.glob("*.npy"))

        if len(self.files) == 0:
            raise RuntimeError(
                f"No .npy ground-truth files found in {self.gt_dir}. "
                f"Please check the dataset path."
            )

        # Verify every GT file has a matching noisy file upfront.
        missing = [
            f.name for f in self.files
            if not (self.noisy_dir / f.name).exists()
        ]
        if missing:
            raise RuntimeError(
                f"Missing NoisyLR files for {len(missing)} ground-truth "
                f"samples, e.g. {missing[:5]}."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        gt_path = self.files[idx]
        noisy_path = self.noisy_dir / gt_path.name

        gt = np.load(gt_path).astype(np.float32)
        noisy = np.load(noisy_path).astype(np.float32)

        gt = torch.from_numpy(gt).unsqueeze(0)
        noisy = torch.from_numpy(noisy).unsqueeze(0)

        if self.augment:
            # --- Data Augmentation (Training Only) ---
            # Apply random horizontal and vertical flips.
            if torch.rand(1).item() < 0.5:  # Horizontal flip
                noisy = torch.flip(noisy, dims=[2])
                gt = torch.flip(gt, dims=[2])
            if torch.rand(1).item() < 0.5:  # Vertical flip
                noisy = torch.flip(noisy, dims=[1])
                gt = torch.flip(gt, dims=[1])

        return noisy, gt
