import os
import shutil
import yaml
from pathlib import Path

# Define how to remap each dataset's class IDs to new class names
# Format: dataset_folder -> {old_class_id: new_class_name}
DATASETS = {
    "datasets/oyster_v2":       {0: "pink_oyster"},
    "datasets/oyster_v1":       {0: "pink_oyster", 1: "pink_oyster", 2: "pink_oyster", 3: "pink_oyster"},
    "datasets/oyster_014":      {0: "oyster_mushroom"},
    "datasets/oyster_04":       {0: "oyster_mushroom"},
    "datasets/oyster_09":       {0: "oyster_mushroom"},
    "datasets/mushroom_general":{0: "king_bolete"},
    "datasets/mushroom_v2":     {1: "mushroom", 2: "mushroom"},  # skip class 0 ('0' label)
}

# Final class list — order matters, this becomes the class ID in merged dataset
FINAL_CLASSES = ["pink_oyster", "oyster_mushroom", "king_bolete", "mushroom"]
class_to_id = {name: i for i, name in enumerate(FINAL_CLASSES)}

OUTPUT = Path("datasets/merged")
for split in ["train", "valid", "test"]:
    (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)

img_count = 0

for dataset_path, class_map in DATASETS.items():
    dataset_path = Path(dataset_path)
    
    for split in ["train", "valid", "test"]:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"
        
        if not img_dir.exists():
            continue
            
        for img_file in img_dir.glob("*.jpg"):
            lbl_file = lbl_dir / (img_file.stem + ".txt")
            if not lbl_file.exists():
                continue
            
            # Read and remap labels
            new_lines = []
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_id = int(parts[0])
                    if old_id not in class_map:
                        continue  # skip unmapped classes
                    new_class_name = class_map[old_id]
                    new_id = class_to_id[new_class_name]
                    new_lines.append(f"{new_id} {' '.join(parts[1:])}")
            
            if not new_lines:
                continue  # skip images with no valid labels
            
            # Copy image and save new label
            new_name = f"{dataset_path.name}_{img_count}"
            shutil.copy(img_file, OUTPUT / split / "images" / f"{new_name}.jpg")
            with open(OUTPUT / split / "labels" / f"{new_name}.txt", "w") as f:
                f.write("\n".join(new_lines))
            img_count += 1

# Write data.yaml
yaml_content = {
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": len(FINAL_CLASSES),
    "names": FINAL_CLASSES
}
with open(OUTPUT / "data.yaml", "w") as f:
    yaml.dump(yaml_content, f)

print(f"Merged {img_count} images into {OUTPUT}")
print(f"Classes: {FINAL_CLASSES}")

# Count per split
for split in ["train", "valid", "test"]:
    count = len(list((OUTPUT / split / "images").glob("*.jpg")))
    print(f"  {split}: {count} images")