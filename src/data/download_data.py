from roboflow import Roboflow
import os

rf = Roboflow(api_key="zRmGEgZ2hjSgN6wLF9qy")
datasets = [
    # Oyster mushroom harvest timing datasets (main ones)
    ("oyster-mushroom-v2", "pink-oyster-harvest-time-2", 1, "datasets/oyster_v2"),
    ("oyster-mushroom-v1", "pink-oyster-harvest-time", 1, "datasets/oyster_v1"),
    ("kritsada-p26-gmail-com", "oyster-mushroom-014", 1, "datasets/oyster_014"),
    ("kritsada-p26-gmail-com", "oyster-mushroom-04", 1, "datasets/oyster_04"),
    ("kritsada-p26-gmail-com", "oyster-mushroom-09", 1, "datasets/oyster_09"),

    # General mushroom detection (teaches model what mushroom looks like)
    ("helge-zu-jeddeloh-l8ugf", "mushroom-detection-p6qkl", 1, "datasets/mushroom_general"),
    ("mark-djeuo", "mushroom-v2-86l4w", 1, "datasets/mushroom_v2"),
]

for workspace, project, version, location in datasets:
    # Skip if already downloaded
    if os.path.exists(location):
        print(f"SKIPPING {location} — already exists")
        continue
    
    print(f"Downloading {project}...")
    try:
        rf.workspace(workspace).project(project).version(version).download(
            "yolov8", location=location
        )
        print(f"Done: {location}")
    except Exception as e:
        print(f"FAILED {project}: {e}")

print("\nAll done!")