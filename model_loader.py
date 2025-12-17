import torch
import timm
import torch.nn as nn
from config import CLASS_NAMES, DEVICE


def load_model(model_path):
    # 1. Create EfficientNet-B0 architecture
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False
    )

    # 2. REBUILD CLASSIFIER EXACTLY AS TRAINED
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, len(CLASS_NAMES))
    )

    # 3. Load checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # 4. Final setup
    model.to(DEVICE)
    model.eval()

    return model
