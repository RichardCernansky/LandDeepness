import os
import json

import cv2
import numpy as np

# Hyperparameters
TARGET_SIZE = 512  # Tile size
STRIDE = 256  # Stride for tiling
RADIUS = 5  # Radius for the mask circle

TRAIN_RATIO = 0.8

# Directories
OBSERVATION_DIR = "./observations"
OUTPUT_DIR = "./preprocessed_observations"
DATASET_DIR = "./dataset"

CENTERS_JSON_FILE = "tree_info.json"

#-----------process-into-tiles------------
def parse_tfw(tfw_path):
    """Parse .tfw file for geospatial transformation information."""
    with open(tfw_path, 'r') as f:
        lines = f.readlines()
    x_scale = float(lines[0].strip())
    y_scale = float(lines[3].strip())
    upper_left_x = float(lines[4].strip())
    upper_left_y = float(lines[5].strip())
    return x_scale, y_scale, upper_left_x, upper_left_y

def generate_mask(tif_path, tfw_path, json_path, output_mask_path):
    """Generate a mask based on tree centers and save as a PNG image."""
    x_scale, y_scale, upper_left_x, upper_left_y = parse_tfw(tfw_path)

    # Load the input image
    tif_image = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    if tif_image is None:
        raise ValueError(f"Unable to read TIF file: {tif_path}")

    mask = np.zeros(tif_image.shape[:2], dtype=np.uint8)  # Create an empty mask

    with open(json_path, 'r') as f:
        center_data = json.load(f)
    center_list = center_data["center_list"]

    for center in center_list:
        longitude, latitude, _ = center
        pixel_x = int((longitude - upper_left_x) / x_scale)
        pixel_y = int((latitude - upper_left_y) / y_scale)

        if 0 <= pixel_x < mask.shape[1] and 0 <= pixel_y < mask.shape[0]:
            cv2.circle(mask, (pixel_x, pixel_y), RADIUS, 255, thickness=-1)

    # Save the mask as a PNG image
    cv2.imwrite(output_mask_path, mask)
    print(f"Mask saved to {output_mask_path}")
    return mask

def generate_tiles_with_opencv(image, output_dir, target_size=512, stride=256, prefix="tile", save_as_png=False):
    """Generate tiles from an input image (TIF or mask) using OpenCV."""
    os.makedirs(output_dir, exist_ok=True)
    k = 0

    for y in range(0, image.shape[0], stride):
        for x in range(0, image.shape[1], stride):
            y_end = min(y + target_size, image.shape[0])
            x_end = min(x + target_size, image.shape[1])

            # Extract tile
            tile = image[y:y_end, x:x_end]

            # Pad tile if necessary
            if tile.shape[0] < target_size or tile.shape[1] < target_size:
                padded_tile = np.zeros((target_size, target_size, tile.shape[2]) if len(tile.shape) == 3 else (target_size, target_size), dtype=tile.dtype)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile

            # Save tile
            ext = "png" if save_as_png else "jpg"
            tile_output_path = os.path.join(output_dir, f"{prefix}_{k}.{ext}")
            cv2.imwrite(tile_output_path, tile)
            k += 1

    print(f"{prefix.capitalize()} tiles saved in {output_dir}")

def process_observations(obs_dir, output_dir):
    """Process all observations in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    observation_folders = [f for f in os.listdir(obs_dir) if os.path.isdir(os.path.join(obs_dir, f))]

    for obs_folder in observation_folders:
        obs_folder_path = os.path.join(obs_dir, obs_folder)
        obs_name = os.path.basename(obs_folder_path)

        # Dynamically name the TIF and TFW files
        tif_file = os.path.join(obs_folder_path, f"{obs_name}.tif")
        tfw_file = os.path.join(obs_folder_path, f"{obs_name}.tfw")
        json_file = os.path.join(obs_folder_path, CENTERS_JSON_FILE)

        if not (os.path.exists(tif_file) and os.path.exists(tfw_file) and os.path.exists(json_file)):
            print(f"Skipping {obs_name}: Missing required files.")
            continue

        obs_output_dir = os.path.join(output_dir, obs_name)
        os.makedirs(obs_output_dir, exist_ok=True)

        # Generate mask
        mask_path = os.path.join(obs_output_dir, "mask.png")
        mask = generate_mask(tif_file, tfw_file, json_file, mask_path)

        # Generate tiles for the original TIF
        tiles_dir = os.path.join(obs_output_dir, "tiles")
        tif_image = cv2.imread(tif_file, cv2.IMREAD_UNCHANGED)
        if tif_image is None:
            raise ValueError(f"Unable to read TIF file: {tif_file}")
        generate_tiles_with_opencv(tif_image, tiles_dir, target_size=TARGET_SIZE, stride=STRIDE, prefix="tif", save_as_png=False)

        # Generate tiles for the mask
        mask_tiles_dir = os.path.join(obs_output_dir, "mask_tiles")
        generate_tiles_with_opencv(mask, mask_tiles_dir, target_size=TARGET_SIZE, stride=STRIDE, prefix="mask", save_as_png=True)

# Run batch processing
process_observations(OBSERVATION_DIR, OUTPUT_DIR)
