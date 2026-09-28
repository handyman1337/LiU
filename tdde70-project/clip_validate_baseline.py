import json
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import open_clip

### Load model
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

model = model.to(device)
model.eval()

### Load Split C
IMAGE_DIR = "RSICD_images"
DATA_C_JSON = "processed_data/dataC.json"

with open(DATA_C_JSON, "r") as f:
    dataC = json.load(f)["images"]

### Build class prompts
class_names = sorted({item["label"] for item in dataC if item["label"] != "Unknown"})

prompts = [f"a satellite image of {name}" for name in class_names]
text_tokens = tokenizer(prompts).to(device)

### Embed prompt
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)

### Dataset for test images
class RSICDImageDataset(Dataset):
    def __init__(self, items, image_dir, preprocess):
        self.items = items
        self.image_dir = Path(image_dir)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = self.image_dir / item["filename"]

        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)

        label = item["label"]
        return image, label, item["filename"]
    
test_dataset = RSICDImageDataset(dataC, IMAGE_DIR, preprocess)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

### Predict with similarity
all_preds = []
all_labels = []
all_filenames = []

with torch.no_grad():
    for images, labels, filenames in test_loader:
        images = images.to(device)

        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        logits = image_features @ text_features.T
        pred_indices = logits.argmax(dim=1)

        preds = [class_names[i] for i in pred_indices.cpu().tolist()]

        all_preds.extend(preds)
        all_labels.extend(labels)
        all_filenames.extend(filenames)

# Accuracy
accuracy = sum(p == y for p, y in zip(all_preds, all_labels)) / len(all_labels)
print(f"Zero-shot accuracy: {accuracy:.4f}")

# Encode labels
class_to_idx = {c: i for i, c in enumerate(class_names)}
y_true = [class_to_idx[y] for y in all_labels]
y_pred = [class_to_idx[p] for p in all_preds]

num_classes = len(class_names)

tp = [0] * num_classes
fp = [0] * num_classes
fn = [0] * num_classes

for t, p in zip(y_true, y_pred):
    tp[t] += (t == p)
    fp[p] += (t != p)
    fn[t] += (t != p)

f1 = 0
for i in range(num_classes):
    prec = tp[i] / (tp[i] + fp[i] + 1e-8)
    rec = tp[i] / (tp[i] + fn[i] + 1e-8)
    f1 += 2 * prec * rec / (prec + rec + 1e-8)

print(f"F1-Score: {f1 / num_classes:.4f}")