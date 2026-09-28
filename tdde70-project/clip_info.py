import torch
import open_clip

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

model = model.to(device)

for p in model.parameters():
    p.requires_grad = False

model.visual.proj.requires_grad = True
model.text_projection.requires_grad = True
model.logit_scale.requires_grad = True


for p in model.visual.transformer.resblocks[-1].parameters():

    p.requires_grad = True

for p in model.transformer.resblocks[-1].parameters():

    p.requires_grad = True

for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)
