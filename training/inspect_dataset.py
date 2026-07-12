from pathlib import Path
from collections import Counter

DATASET_ROOT = Path("datasets/raw/archive")
splits = ["train","val","test"]
print ("=" * 60)
print("MEDVISION DATASET INSPECTION")
print("=" * 60)
grand_total = 0
for split in splits:
    split_path = DATASET_ROOT/split

    print(f"\n{split.upper()} SET")
    print("-" * 60)
    total = 0

    for class_dir in sorted(split_path.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*")))
            total += count

            print(f"{class_dir.name:<15}:{count}")
    print(f"\nTotal images:{total}")
    grand_total += total
print("\n" + "="*60)
print(f"Total dataset size:{grand_total}")
