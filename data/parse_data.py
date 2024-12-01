import os
import json
import glob
import cv2
import numpy as np

# Parent directory containing observation folders
OBSERVATION_DIR = "./observations"
OUTPUT_DIR = "./processed_observations"

# Target size for tiling segments
TARGET_SIZE = 512

# Function to parse .tfw file
def parse_tfw(tfw_path):
    with open(tfw_path, 'r') as f:
        lines = f.readlines()
    x_scale = float(lines[0].strip())
    y_scale = float(lines[3].strip())
    upper_left_x = float(lines[4].strip())
    upper_left_y = float(lines[5].strip())
    return x_scale, y_scale, upper_left_x, upper_left_y

# Function to generate mask from JSON center_list
def generate_mask(tif_path, tfw_path, json_path, output_mask_path):
    x_scale, y_scale, upper_left_x, upper_left_y = parse_tfw(tfw_path)
    tif_image = cv2.imread(tif_path)
    mask = np.zeros(tif_image.shape[:2], dtype=np.uint8)

    with open(json_path, 'r') as f:
        center_data = json.load(f)
    center_list = center_data["center_list"]

    for center in center_list:
        longitude, latitude, _ = center
        pixel_x = int((longitude - upper_left_x) / x_scale)
        pixel_y = int((latitude - upper_left_y) / y_scale)

        if 0 <= pixel_x < mask.shape[1] and 0 <= pixel_y < mask.shape[0]:
            mask[pixel_y, pixel_x] = 255  # Mark center as white

    cv2.imwrite(output_mask_path, mask)
    print(f"Mask saved to {output_mask_path}")

# Function to generate tiled segments
def generate_segments(tif_path, mask_path, output_dir):
    img = cv2.imread(tif_path)
    mask = cv2.imread(mask_path)

    os.makedirs(output_dir, exist_ok=True)
    k = 0
    for y in range(0, img.shape[0], TARGET_SIZE):
        for x in range(0, img.shape[1], TARGET_SIZE):
            img_tile = img[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
            mask_tile = mask[y:y + TARGET_SIZE, x:x + TARGET_SIZE]

            if img_tile.shape[0] == TARGET_SIZE and img_tile.shape[1] == TARGET_SIZE:
                out_img_path = os.path.join(output_dir, f"tile_{k}_img.jpg")
                out_mask_path = os.path.join(output_dir, f"tile_{k}_mask.png")
                cv2.imwrite(out_img_path, img_tile)
                cv2.imwrite(out_mask_path, mask_tile)
                k += 1

    print(f"Segments saved in {output_dir}")

# Main batch processing function
def process_observations(obs_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    observation_folders = glob.glob(os.path.join(obs_dir, "*"))

    for obs_folder in observation_folders:
        obs_name = os.path.basename(obs_folder)
        tif_path = os.path.join(obs_folder, "image.tif")
        tfw_path = os.path.join(obs_folder, "image.tfw")
        json_path = os.path.join(obs_folder, "centers.json")

        if not (os.path.exists(tif_path) and os.path.exists(tfw_path) and os.path.exists(json_path)):
            print(f"Skipping {obs_name}: Missing required files.")
            continue

        obs_output_dir = os.path.join(output_dir, obs_name)
        os.makedirs(obs_output_dir, exist_ok=True)

        # Generate mask
        mask_path = os.path.join(obs_output_dir, "mask.png")
        generate_mask(tif_path, tfw_path, json_path, mask_path)

        # Generate segments
        segments_dir = os.path.join(obs_output_dir, "segments")
        generate_segments(tif_path, mask_path, segments_dir)

# Run batch processing
process_observations(OBSERVATION_DIR, OUTPUT_DIR)
