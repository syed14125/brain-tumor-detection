import cv2
import numpy as np

def overlay_cam(image, cam):
    image = np.array(image.resize((224,224))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap / 255.0
    overlay = 0.6 * image + 0.4 * heatmap
    return np.clip(overlay, 0, 1)
