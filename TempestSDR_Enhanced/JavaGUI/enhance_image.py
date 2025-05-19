import os.path
import sys
import cv2
import numpy as np
import torch
from datetime import datetime
from utils import utils_image as util


def perform_inference(model, device, model_type, input_path, output_path, width, height):
    if model_type.lower() == "drunet":
        img_L_original = util.imread_uint(input_path, n_channels=3)
        img_L_original = extract_exact_patch(img_L_original, width, height)
        img_L = img_L_original[:, :, :2]
        img_L = util.uint2single(img_L)
        img_L = util.single2tensor4(img_L).to(device)

    elif model_type.lower() == "dncnn":
        img_L_original = util.imread_uint(input_path, n_channels=1)
        img_L_original = extract_exact_patch(img_L_original, width, height)
        img_L = util.uint2single(img_L_original)
        img_L = util.single2tensor4(img_L).to(device)

    else:
        print(
            f"[ERROR] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} I do not know this model type {model_type}. Please use DRUNet or DnCNN.")
        sys.exit(1)

    with torch.no_grad():
        img_E = model(img_L)
    img_E = util.tensor2uint(img_E)

    util.imsave(img_E, output_path)
    print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} Inference complete.", flush=True)
    sys.exit(0)


def load_model(model_type, model_path):
    try:
        if model_type.lower() == "drunet":
            from models.network_unet import UNetRes as net
            model = net(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', bias=False)
        elif model_type.lower() == "dncnn":
            from models.network_dncnn import DnCNN as net
            model = net(in_nc=1, out_nc=1, nc=64, nb=17, act_mode="BR")
        else:
            print(
                f"[ERROR] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} I do not know this model type {model_type}. Please use DRUNet or DnCNN.",
                flush=True)
            sys.exit(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

        print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} Model {model_type} loaded from {model_path}.",
              flush=True)
        return model, device
    except Exception as e:
        print(f"[ERROR] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} Loading model failed due to {e}.", flush=True)
        sys.exit(1)


def extract_exact_patch(img, target_width, target_height):
    try:
        edges = cv2.Canny(img, 30, 100)

        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError("Contour not found.")

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        if w < target_width or h < target_height:
            raise ValueError(f"Detected region ({w}x{h}) is smaller than target {target_width}x{target_height}.")

        x_crop = x + (w - target_width) // 2
        y_crop = y + (h - target_height) // 2
        cropped = img[y_crop:y_crop + target_height, x_crop:x_crop + target_width]

        if cropped.shape[0] != target_height or cropped.shape[1] != target_width:
            raise ValueError("Final cropped region doesn't match target resolution.")

        return cropped
    except Exception as e:
        print(f"[WARNING] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} Cropping contour logic failed due to: {e}",
              flush=True)
        h, w = img.shape[:2]
        x_center = w // 2
        y_center = h // 2
        x_crop = x_center - target_width // 2
        y_crop = y_center - target_height // 2
        cropped = img[y_crop:y_crop + target_height, x_crop:x_crop + target_width]
        print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} Fallback: Cropped from center.", flush=True)
        return cropped


def main():
    if len(sys.argv) != 6:
        print(
            f"[ERROR] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} Usage: python enhance_image.py input_path output_path model_type model_path target_resolution",
            flush=True)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    model_type = sys.argv[3]
    model_path = sys.argv[4]
    target_resolution = sys.argv[5]

    print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} Input path: " + input_path, flush=True)
    print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} Outputh path: " + output_path, flush=True)
    print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} Model type: " + model_type, flush=True)
    print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} Model path: " + model_path, flush=True)
    print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} Target resolution: " + target_resolution,
          flush=True)

    resolution_only = target_resolution.split("@")[0].strip()
    width_str, height_str = resolution_only.split("x")
    width = int(width_str)
    height = int(height_str)

    print(
        f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} Parsed resolution: width = {width}, height = {height}",
        flush=True)

    model, device = load_model(model_type, model_path)

    perform_inference(model, device, model_type, input_path, output_path, width, height)


if __name__ == '__main__':
    main()
