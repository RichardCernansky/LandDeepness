import os
import json
import glob
import cv2
import numpy as np
from osgeo import gdal

#hyperparms list:
    #target_size
    #stride
    #radius (of the masking circle)

# parent directory containing observation folders
OBSERVATION_DIR = "./observations"
OUTPUT_DIR = "./processed_observations"

GSDDSM_TIF_FILE = "gsddsm.tif"
GSDDSM_TFW_FILE = "gsddsm.tfw"
CENTERS_JSON_FILE = "tree_info.json"

def read_tif_with_gdal(tif_path):
    dataset = gdal.Open(tif_path)
    if not dataset:
        raise ValueError(f"Unable to read TIF file: {tif_path}")
    return dataset.ReadAsArray()

# function to parse .tfw file
def parse_tfw(tfw_path: str):
    with open(tfw_path, 'r') as f:
        lines = f.readlines()
    x_scale = float(lines[0].strip())
    y_scale = float(lines[3].strip())
    upper_left_x = float(lines[4].strip())
    upper_left_y = float(lines[5].strip())
    return x_scale, y_scale, upper_left_x, upper_left_y

# function to generate mask from JSON center_list
def generate_mask(tif_path: str, tfw_path: str, json_path: str, output_mask_path: str):
    x_scale, y_scale, upper_left_x, upper_left_y = parse_tfw(tfw_path)
    tif_image = read_tif_with_gdal(tif_path)
    mask = np.zeros(tif_image.shape[:2], dtype=np.uint8)

    with open(json_path, 'r') as f:
        center_data = json.load(f)
    center_list = center_data["center_list"]

    radius = 5 #hyperparameter
    for center in center_list:
        longitude, latitude, _ = center
        pixel_x = int((longitude - upper_left_x) / x_scale)
        pixel_y = int((latitude - upper_left_y) / y_scale)

        if 0 <= pixel_x < mask.shape[1] and 0 <= pixel_y < mask.shape[0]:
            cv2.circle(mask, (pixel_x, pixel_y), radius, 255, thickness=-1)

    cv2.imwrite(output_mask_path, mask)
    print(f"Mask saved to {output_mask_path}")


# function to generate tiled segments with overlap
def generate_segments_overlap(tif_path: str, mask_path: str, output_dir: str, target_size=512, stride=256):
    """
    generate tiled segments with overlap from the input image and mask.
    for now, it ignores the remaining pixels at the edges (in the last iteration)
    args:
        tif_path - path to the input image (TIF file).
        mask_path - path to the corresponding mask.
        output_dir - directory where the segments will be saved.
        target_size - size of each tile.
        stride - step size for sliding window (overlap if stride < target_size).
    """
    img = read_tif_with_gdal(tif_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # read mask as grayscale

    os.makedirs(output_dir, exist_ok=True)
    k = 0
    for y in range(0, img.shape[0] - target_size + 1, stride):
        for x in range(0, img.shape[1] - target_size + 1, stride):
            img_tile = img[y:y + target_size, x:x + target_size]
            mask_tile = mask[y:y + target_size, x:x + target_size]

            if img_tile.shape[0] == target_size and img_tile.shape[1] == target_size:
                out_img_path = os.path.join(output_dir, f"tile_{k}_img.jpg")
                out_mask_path = os.path.join(output_dir, f"tile_{k}_mask.png")
                cv2.imwrite(out_img_path, img_tile)
                cv2.imwrite(out_mask_path, mask_tile)
                k += 1

    print(f"Segments with overlap saved in {output_dir}")

# main batch processing function
def process_observations(obs_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    observation_folders = glob.glob(os.path.join(obs_dir, "*"))

    for obs_folder in observation_folders:
        obs_name = os.path.basename(obs_folder)
        tif_path = os.path.join(obs_folder, GSDDSM_TIF_FILE)
        tfw_path = os.path.join(obs_folder, GSDDSM_TFW_FILE)
        json_path = os.path.join(obs_folder, CENTERS_JSON_FILE)

        if not (os.path.exists(tif_path) and os.path.exists(tfw_path) and os.path.exists(json_path)):
            print(f"Skipping {obs_name}: Missing required files.")
            continue

        obs_output_dir = os.path.join(output_dir, obs_name)
        os.makedirs(obs_output_dir, exist_ok=True)

        # generate mask
        mask_path = os.path.join(obs_output_dir, "mask.png")
        generate_mask(tif_path, tfw_path, json_path, mask_path)

        # generate segments
        segments_dir = os.path.join(obs_output_dir, "segments")
        generate_segments_overlap(tif_path, mask_path, segments_dir)

# run batch processing
process_observations(OBSERVATION_DIR, OUTPUT_DIR)
