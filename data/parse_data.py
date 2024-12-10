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

# Parse TFW file
def parse_tfw(tfw_path):
    """Parse .tfw file for geospatial transformation information."""
    with open(tfw_path, 'r') as f:
        lines = f.readlines()
    x_scale = float(lines[0].strip())
    y_scale = float(lines[3].strip())
    upper_left_x = float(lines[4].strip())
    upper_left_y = float(lines[5].strip())
    return x_scale, y_scale, upper_left_x, upper_left_y

# Generate mask
def generate_mask(tif_path, tfw_path, json_path, output_mask_path):
    """Generate a mask based on tree centers and save as a PNG image."""
    x_scale, y_scale, upper_left_x, upper_left_y = parse_tfw(tfw_path)

    tif_image = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    if tif_image is None:
        raise ValueError(f"Unable to read TIF file: {tif_path}")

    mask = np.zeros(tif_image.shape[:2], dtype=np.uint8)

    with open(json_path, 'r') as f:
        center_data = json.load(f)
    center_list = center_data["center_list"]

    for center in center_list:
        longitude, latitude, _ = center
        pixel_x = int((longitude - upper_left_x) / x_scale)
        pixel_y = int((latitude - upper_left_y) / y_scale)

        if 0 <= pixel_x < mask.shape[1] and 0 <= pixel_y < mask.shape[0]:
            cv2.circle(mask, (pixel_x, pixel_y), RADIUS, 255, thickness=-1)

    cv2.imwrite(output_mask_path, mask)
    print(f"Mask saved to {output_mask_path}")
    return mask

# Generate YOLO labels from mask
def generate_yolo_labels(mask, image_tile_name, label_output_path, target_size=512):
    """Generate YOLO labels for a single image tile."""
    tile_labels = []

    # Find contours and bounding boxes in the mask tile
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x_min, y_min, w, h = cv2.boundingRect(contour)
        x_max, y_max = x_min + w, y_min + h
        x_center = (x_min + x_max) / 2 / target_size
        y_center = (y_min + y_max) / 2 / target_size
        box_width = w / target_size
        box_height = h / target_size

        # Add YOLO label (class_id is 0 for trees)
        tile_labels.append(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    # Save labels with the same name as the image tile
    label_file = os.path.join(label_output_path, f"{image_tile_name}.txt")
    with open(label_file, 'w') as f:
        f.write("\n".join(tile_labels))

# Generate tiles
def generate_tiles_with_labels(image, mask, output_dir, label_dir, mask_tiles_dir, target_size=512, stride=256, prefix="tile", save_as_png=False):
    """Generate tiles for both the TIF image and the mask, and corresponding YOLO labels."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(mask_tiles_dir, exist_ok=True)
    k = 0

    for y in range(0, image.shape[0], stride):
        for x in range(0, image.shape[1], stride):
            y_end = min(y + target_size, image.shape[0])
            x_end = min(x + target_size, image.shape[1])

            # Extract image tile
            image_tile = image[y:y_end, x:x_end]
            tile_name = f"{prefix}_{k:05d}"

            # Pad image tile if necessary
            if image_tile.shape[0] < target_size or image_tile.shape[1] < target_size:
                padded_tile = np.zeros((target_size, target_size, image_tile.shape[2]) if len(image_tile.shape) == 3 else (target_size, target_size), dtype=image_tile.dtype)
                padded_tile[:image_tile.shape[0], :image_tile.shape[1]] = image_tile
                image_tile = padded_tile

            # Save image tile
            ext = "png" if save_as_png else "jpg"
            tile_output_path = os.path.join(output_dir, f"{tile_name}.{ext}")
            cv2.imwrite(tile_output_path, image_tile)

            # Extract mask tile
            mask_tile = mask[y:y_end, x:x_end]
            if mask_tile.shape[0] < target_size or mask_tile.shape[1] < target_size:
                padded_mask = np.zeros((target_size, target_size), dtype=mask.dtype)
                padded_mask[:mask_tile.shape[0], :mask_tile.shape[1]] = mask_tile
                mask_tile = padded_mask

            # Save mask tile
            mask_tile_output_path = os.path.join(mask_tiles_dir, f"{tile_name}.png")
            cv2.imwrite(mask_tile_output_path, mask_tile)

            # Generate YOLO labels for the mask tile
            generate_yolo_labels(mask_tile, tile_name, label_dir, target_size)

            k += 1

    print(f"Tiles, mask tiles, and labels saved in {output_dir}, {mask_tiles_dir}, and {label_dir}")

# Process observations
def process_observations(obs_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    observation_folders = [f for f in os.listdir(obs_dir) if os.path.isdir(os.path.join(obs_dir, f))]

    for obs_folder in observation_folders:
        obs_folder_path = os.path.join(obs_dir, obs_folder)
        obs_name = os.path.basename(obs_folder_path)

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

        # Generate tiles for the original TIF and mask, along with YOLO labels
        tiles_dir = os.path.join(obs_output_dir, "tiles")
        labels_dir = os.path.join(obs_output_dir, "labels")
        mask_tiles_dir = os.path.join(obs_output_dir, "mask_tiles")
        tif_image = cv2.imread(tif_file, cv2.IMREAD_UNCHANGED)
        generate_tiles_with_labels(tif_image, mask, tiles_dir, labels_dir, mask_tiles_dir, target_size=TARGET_SIZE, stride=STRIDE, prefix="tile", save_as_png=False)


# Run batch processing
process_observations(OBSERVATION_DIR, OUTPUT_DIR)
