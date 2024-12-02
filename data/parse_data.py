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


def save_tile_as_tif(tile, output_path, geo_transform, projection):
    """
    Save a single tile as a Float32 GeoTIFF using GDAL.
    """
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = tile.shape
    dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)

    band = dataset.GetRasterBand(1)
    band.WriteArray(tile)
    band.SetNoDataValue(-9999)  # Optional: Set no-data value if needed
    dataset.FlushCache()

def generate_float32_tiles(tif_path, output_dir, target_size=512, stride=256):
    """
    Generate tiles as 32-bit Float GeoTIFFs with overlap.
    """
    dataset = gdal.Open(tif_path)
    img = dataset.ReadAsArray()
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    os.makedirs(output_dir, exist_ok=True)
    k = 0

    for y in range(0, img.shape[0], stride):
        for x in range(0, img.shape[1], stride):
            y_end = min(y + target_size, img.shape[0])
            x_end = min(x + target_size, img.shape[1])

            tile = img[y:y_end, x:x_end]

            # Update geotransform for the tile
            tile_geo_transform = (
                geo_transform[0] + x * geo_transform[1],
                geo_transform[1],
                geo_transform[2],
                geo_transform[3] + y * geo_transform[5],
                geo_transform[4],
                geo_transform[5]
            )

            # Save the tile
            tile_output_path = os.path.join(output_dir, f"tile_{k}.tif")
            save_tile_as_tif(tile, tile_output_path, tile_geo_transform, projection)
            k += 1

    print(f"Float32 tiles saved in {output_dir}")

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
        generate_float32_tiles(tif_path, segments_dir)

# run batch processing
process_observations(OBSERVATION_DIR, OUTPUT_DIR)
