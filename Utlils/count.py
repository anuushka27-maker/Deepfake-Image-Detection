# count_images.py
import os

root = "/mnt/c/Deepfake Image Detection/Final_Dataset"
splits = ["train", "validation", "test"]
for s in splits:
    print(f"=== {s} ===")
    split_dir = os.path.join(root, s)
    if not os.path.isdir(split_dir):
        print("  missing:", split_dir)
        continue
    for cls in sorted(os.listdir(split_dir)):
        cls_path = os.path.join(split_dir, cls)
        if os.path.isdir(cls_path):
            n = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
            print(f"  {cls}: {n}")
    print()