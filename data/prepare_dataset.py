import shutil
import itertools
import random
import os
from concurrent.futures import ThreadPoolExecutor

# Parameters
TRAIN_RATIO = 0.8
TEST_RATIO = 0.1  # Remaining will be used for validation
OUTPUT_DIR = "./preprocessed_observations"
DATASET_DIR = "./dataset"

# Helper Functions
def chunk_iterator(iterator, chunk_size):
    """Yield successive chunks from the iterator."""
    chunk = list(itertools.islice(iterator, chunk_size))
    while chunk:
        yield chunk
        chunk = list(itertools.islice(iterator, chunk_size))

def copy_file_group(group, dest_images_dir, dest_masks_dir, dest_labels_dir):
    """Copy a file group (tile, mask, label) to the destination directories."""
    tile_path, mask_path, label_path = group
    shutil.copy(tile_path, os.path.join(dest_images_dir, os.path.basename(tile_path)))
    shutil.copy(mask_path, os.path.join(dest_masks_dir, os.path.basename(mask_path)))
    shutil.copy(label_path, os.path.join(dest_labels_dir, os.path.basename(label_path)))

def prepare_dataset(obs_dir, dataset_dir, train_ratio=0.8, test_ratio=0.1, chunk_size=10000, workers=4):
    """Prepare the dataset with train, validation, and test splits."""
    # Create directories for train, val, and test splits
    dirs = {
        "train": {
            "images": os.path.join(dataset_dir, "train/images"),
            "masks": os.path.join(dataset_dir, "train/masks"),
            "labels": os.path.join(dataset_dir, "train/labels"),
        },
        "val": {
            "images": os.path.join(dataset_dir, "val/images"),
            "masks": os.path.join(dataset_dir, "val/masks"),
            "labels": os.path.join(dataset_dir, "val/labels"),
        },
        "test": {
            "images": os.path.join(dataset_dir, "test/images"),
            "masks": os.path.join(dataset_dir, "test/masks"),
            "labels": os.path.join(dataset_dir, "test/labels"),
        },
    }

    for split, split_dirs in dirs.items():
        for sub_dir in split_dirs.values():
            os.makedirs(sub_dir, exist_ok=True)

    # Collect all file groups (tiles, masks, labels)
    obs_folders = [f for f in os.listdir(obs_dir) if os.path.isdir(os.path.join(obs_dir, f))]
    file_groups = (
        (
            os.path.join(tiles_dir, tile_file),
            os.path.join(mask_tiles_dir, tile_file.replace(".jpg", ".png")),
            os.path.join(labels_dir, tile_file.replace(".jpg", ".txt")),
        )
        for obs_folder in obs_folders
        for tiles_dir in [os.path.join(obs_dir, obs_folder, "tiles")]
        for mask_tiles_dir in [os.path.join(obs_dir, obs_folder, "mask_tiles")]
        for labels_dir in [os.path.join(obs_dir, obs_folder, "labels")]
        for tile_file in os.listdir(tiles_dir)
        if os.path.exists(os.path.join(mask_tiles_dir, tile_file.replace(".jpg", ".png")))
        and os.path.exists(os.path.join(labels_dir, tile_file.replace(".jpg", ".txt")))
    )

    # Shuffle and split the dataset
    file_groups = list(file_groups)  # Materialize the generator for shuffling and splitting
    random.shuffle(file_groups)

    total_files = len(file_groups)
    train_count = int(total_files * train_ratio)
    test_count = int(total_files * test_ratio)
    val_count = total_files - train_count - test_count

    train_groups = file_groups[:train_count]
    test_groups = file_groups[train_count:train_count + test_count]
    val_groups = file_groups[train_count + test_count:]

    # Helper function to copy groups in parallel
    def copy_groups(groups, dest_dirs):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            executor.map(
                lambda group: copy_file_group(
                    group,
                    dest_dirs["images"],
                    dest_dirs["masks"],
                    dest_dirs["labels"],
                ),
                groups,
            )

    # Process each split
    copy_groups(train_groups, dirs["train"])
    copy_groups(test_groups, dirs["test"])
    copy_groups(val_groups, dirs["val"])

    print(f"Dataset prepared: {len(train_groups)} training samples, {len(val_groups)} validation samples, {len(test_groups)} test samples.")

# Run the dataset preparation
prepare_dataset(OUTPUT_DIR, DATASET_DIR, train_ratio=TRAIN_RATIO, test_ratio=TEST_RATIO)
