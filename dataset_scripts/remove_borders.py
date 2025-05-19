import cv2
import numpy as np
import os

CAPTURED_FOLDER = ""
REFERENCE_FOLDER = ""
OUTPUT_FOLDER = ""

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def align_and_crop(captured_path, reference_path):
    captured = cv2.imread(captured_path)
    reference = cv2.imread(reference_path)

    if captured is None or reference is None:
        print(f"Error loading images: {captured_path}, {reference_path}")
        return None

    captured_gray = cv2.cvtColor(captured, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(captured_gray, reference_gray, cv2.TM_CCORR)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < 0.3:
        print(f"Poor match for {captured_path}, skipping.")
        return None

    x, y = max_loc
    h, w = reference.shape[:2]
    cropped = captured[y:y+h, x:x+w]

    return cropped

for filename in os.listdir(CAPTURED_FOLDER):
    if filename.endswith(".png") and filename.endswith("_5.png"):
        base_name = filename[:-6]

        captured_path = os.path.join(CAPTURED_FOLDER, filename)
        reference_path = os.path.join(REFERENCE_FOLDER, base_name + ".png")

        if os.path.exists(reference_path):
            cropped_image = align_and_crop(captured_path, reference_path)
            if cropped_image is not None:
                output_path = os.path.join(OUTPUT_FOLDER, filename)
                cv2.imwrite(output_path, cropped_image)
                print(f"Processed: {filename}")
        else:
            print(f"No matching reference found for {filename}, skipping.")

print("Processing complete. Cropped images saved in:", OUTPUT_FOLDER)