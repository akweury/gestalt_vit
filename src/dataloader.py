# Created by MacBook Pro at 13.04.25
from torch.utils.data import DataLoader
from src.dataset import GestaltDataset
from src.transform_adaptive_patch import AdaptivePatcher

def get_gestalt_loader(root_dir, principle, split='train', batch_size=8, shuffle=True, num_workers=2):
    transform = AdaptivePatcher(threshold=0.08, min_size=16, output_size=(16, 16))

    dataset = GestaltDataset(
        root_dir=root_dir,
        principle=principle,
        split=split,
        transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)