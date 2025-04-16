# Created by MacBook Pro at 16.04.25

import os
import shutil

val_dir = "/tiny-224/val"
img_dir = os.path.join(val_dir, "images")
ann_file = os.path.join(val_dir, "val_annotations.txt")

with open(ann_file, 'r') as f:
    for line in f:
        fname, cname = line.strip().split('\t')[:2]
        class_dir = os.path.join(val_dir, cname)
        os.makedirs(class_dir, exist_ok=True)
        shutil.move(os.path.join(img_dir, fname), os.path.join(class_dir, fname))

shutil.rmtree(img_dir)
print("✅ 验证集整理完成。")