import json
from pathlib import Path

# Paths
INPUT_JSON = "dataset_rsicd.json"
TXT_CLASSES_DIR = "txtclasses_rsicd"
OUTPUT_DIR = "processed_data"

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load original dataset
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

# Build filename → label mapping from txt files
filename_to_label = {}
for txt_file in Path(TXT_CLASSES_DIR).glob("*.txt"):
    label = txt_file.stem  # 'Airport', 'BareLand', etc.
    with open(txt_file, "r") as f:
        for line in f:
            filename = line.strip()
            if filename:
                filename_to_label[filename] = label

# Extract image entries
images = data["images"]

# Split datasets
dataA = []
dataB = []
dataC = []

for item in images:
    split = item["split"]
    
    # Add the label key
    filename = item["filename"]
    item["label"] = filename_to_label.get(filename, "Unknown")

    if split == "train":
        dataA.append(item)
    elif split == "val":
        dataB.append(item)
    elif split == "test":
        dataC.append(item)

# Save Split A
with open(f"{OUTPUT_DIR}/dataA.json", "w") as f:
    json.dump({"images": dataA}, f, indent=4)

# Save Split B
with open(f"{OUTPUT_DIR}/dataB.json", "w") as f:
    json.dump({"images": dataB}, f, indent=4)

# Save Split C
with open(f"{OUTPUT_DIR}/dataC.json", "w") as f:
    json.dump({"images": dataC}, f, indent=4)

print(f"Saved {len(dataA)} samples to dataA.json")
print(f"Saved {len(dataB)} samples to dataB.json")
print(f"Saved {len(dataC)} samples to dataC.json")