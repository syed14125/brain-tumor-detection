import torch

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224
