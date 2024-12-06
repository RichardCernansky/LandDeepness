import shutil
import itertools
import random
import os

from concurrent.futures import ThreadPoolExecutor

TRAIN_RATIO = 0.8
OUTPUT_DIR = "./preprocessed_observations"
DATASET_DIR = "./dataset"

#-----------sort-out-into-training-data------------
def chunk_iterator(iterator, chunk_size):
    """Yield successive chunks from the iterator."""
    chunk = list(itertools.islice(iterator, chunk_size))
    while chunk:
        yield chunk
        chunk = list(itertools.islice(iterator, chunk_size))

def copy_file_pair(pair, dest_images_dir, dest_masks_dir):
    """Copy a file pair to the destination directories."""
    tile_path, mask_path = pair
    shutil.copy(tile_path, os.path.join(dest_images_dir, os.path.basename(tile_path)))
    shutil.copy(mask_path, os.path.join(dest_masks_dir, os.path.basename(mask_path)))

def prepare_dataset_hybrid(obs_dir, dataset_dir, train_ratio=0.8, chunk_size=10000, workers=4):
    """Prepare the dataset using chunked and parallel processing."""
    train_images_dir = os.path.join(dataset_dir, "train/images")
    train_masks_dir = os.path.join(dataset_dir, "train/masks")
    val_images_dir = os.path.join(dataset_dir, "val/images")
    val_masks_dir = os.path.join(dataset_dir, "val/masks")

    for sub_dir in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir]:
        os.makedirs(sub_dir, exist_ok=True)

    obs_folders = [f for f in os.listdir(obs_dir) if os.path.isdir(os.path.join(obs_dir, f))]

    file_pairs = (
        (os.path.join(tiles_dir, tile_file), os.path.join(mask_tiles_dir, tile_file.replace("tif_", "mask_").replace(".jpg", ".png")))
        for obs_folder in obs_folders
        for tiles_dir in [os.path.join(obs_dir, obs_folder, "tiles")]
        for mask_tiles_dir in [os.path.join(obs_dir, obs_folder, "mask_tiles")]
        for tile_file in os.listdir(tiles_dir)
        if os.path.exists(os.path.join(mask_tiles_dir, tile_file.replace("tif_", "mask_").replace(".jpg", ".png")))
    )

    total_train, total_val = 0, 0

    for chunk in chunk_iterator(file_pairs, chunk_size):
        random.shuffle(chunk)  # Shuffle within the chunk

        train_split = int(len(chunk) * train_ratio)
        train_pairs = chunk[:train_split]
        val_pairs = chunk[train_split:]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            executor.map(lambda pair: copy_file_pair(pair, train_images_dir, train_masks_dir), train_pairs)
            executor.map(lambda pair: copy_file_pair(pair, val_images_dir, val_masks_dir), val_pairs)

        total_train += len(train_pairs)
        total_val += len(val_pairs)

    print(f"Dataset prepared: {total_train} training samples, {total_val} validation samples.")


prepare_dataset_hybrid(OUTPUT_DIR, DATASET_DIR, train_ratio=TRAIN_RATIO)
