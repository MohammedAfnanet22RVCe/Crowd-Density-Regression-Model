# Global configuration for Crowd Density Regression project

IMAGE_SIZE = (995, 421)   # (height, width)
EPOCHS = 100
BATCH_SIZE = 4
VAL_SPLIT = 0.2

DATASET_PATH = "data/ShanghaiTech/part_B"
TRAIN_IMG_DIR = f"{DATASET_PATH}/train_data/images"
TRAIN_GT_DIR = f"{DATASET_PATH}/train_data/ground-truth"

MODEL_SAVE_PATH = "models/resized_model_small.h5"
