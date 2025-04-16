# Created by MacBook Pro at 16.04.25


import os
import shutil
from pathlib import Path

def build_subset(source_dir, target_dir, num_classes=5):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    splits = ['train', 'val']

    for split in splits:
        src_split = source_dir / split
        dst_split = target_dir / split
        dst_split.mkdir(parents=True, exist_ok=True)

        # 获取类名（子文件夹名），排序后取前 N 类
        class_dirs = sorted([d for d in src_split.iterdir() if d.is_dir()])[:num_classes]
        print(f"Processing {split}: {len(class_dirs)} classes")

        for cls_dir in class_dirs:
            dst_cls = dst_split / cls_dir.name
            shutil.copytree(cls_dir, dst_cls)
            print(f"  -> Copied {cls_dir.name}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build Tiny ImageNet subset')
    parser.add_argument('--source', type=str, required=True, help='Path to original tiny-imagenet-200')
    parser.add_argument('--target', type=str, required=True, help='Output directory for subset')
    parser.add_argument('--num-classes', type=int, default=5, help='Number of classes to include')

    args = parser.parse_args()

    build_subset(args.source, args.target, args.num_classes)