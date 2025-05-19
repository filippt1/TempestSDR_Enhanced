import os
import numpy as np
import cv2
import pytesseract
import fastwer
from skimage.metrics import structural_similarity as ssim
from utils import utils_image as util

# Script that evaluates the performance of image reconstruction. Computes MSE, PSNR, SSIM and CER.

# Define the path to directory with reference images and its captures.
dataroot_L = ""

def compute_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def extract_text(img):
    return pytesseract.image_to_string(img).strip().replace("\n", " ")

def compute_cer(gt_text, est_text):
    return fastwer.score([est_text], [gt_text], char_level=True) / 100



mse_list, psnr_list, ssim_list, cer_list = [], [], [], []

for gt_file in os.listdir(dataroot_L):
    if not gt_file.endswith(".png"):
        continue

    gt_path = os.path.join(dataroot_L, gt_file)
    img_name, _ = os.path.splitext(gt_file)

    estimate_dir = os.path.join(dataroot_L, img_name)
    if not os.path.isdir(estimate_dir):
        print(f"Skipping {img_name} (No estimates found)")
        continue

    gt_img = util.imread_uint(gt_path, n_channels=3)
    if gt_img.ndim == 3:
        gt_img = np.mean(gt_img, axis=2)
        gt_img = gt_img.astype('uint8')

    gt_text = extract_text(gt_img)
    mse_list_temp, psnr_list_temp, ssim_list_temp, cer_list_temp = [], [], [], []

    for estimate_file in os.listdir(estimate_dir):
        if not estimate_file.endswith(".png"):
            continue

        estimate_path = os.path.join(estimate_dir, estimate_file)
        est_img = util.imread_uint(estimate_path, n_channels=1)
        est_img = est_img[:, :, 0]
        est_text = extract_text(est_img)

        mse_val = compute_mse(gt_img, est_img)
        psnr_val = util.calculate_psnr(gt_img, est_img)
        ssim_val = util.calculate_ssim(gt_img, est_img)
        cer_val = compute_cer(gt_text, est_text)

        mse_list_temp.append(mse_val)
        psnr_list_temp.append(psnr_val)
        ssim_list_temp.append(ssim_val)
        cer_list_temp.append(cer_val)

    print(f"{gt_file}: MSE={np.mean(mse_list_temp):.4f}, PSNR={np.mean(psnr_list_temp):.2f}, SSIM={np.mean(ssim_list_temp):.4f}, CER={np.mean(cer_list_temp):.4f}")
    mse_list.append(np.mean(mse_list_temp))
    psnr_list.append(np.mean(psnr_list_temp))
    ssim_list.append(np.mean(ssim_list_temp))
    cer_list.append(np.mean(cer_list_temp))

avg_mse = np.mean(mse_list)
avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)
avg_cer = np.mean(cer_list)

print("\nFinal Averages:")
print(f"Avg MSE: {avg_mse:.4f}")
print(f"Avg PSNR: {avg_psnr:.2f}")
print(f"Avg SSIM: {avg_ssim:.4f}")
print(f"Avg CER: {avg_cer:.4f}")