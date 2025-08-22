import os
import cv2
import scipy.io
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from .config import IMAGE_SIZE, TRAIN_IMG_DIR, TRAIN_GT_DIR

def load_data(img_dir, gt_dir):
    X, y = [], []
    img_files = sorted(os.listdir(img_dir))

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        mat_file = f"GT_{img_file.split('.')[0]}.mat"
        mat_path = os.path.join(gt_dir, mat_file)

        if not os.path.exists(mat_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        X.append(img)

        mat = scipy.io.loadmat(mat_path)
        annPoints = mat["image_info"][0][0][0][0][0]
        y.append(len(annPoints))  # crowd count

    return np.array(X), np.array(y)

def preprocess_data(X, y):
    X_resized = tf.image.resize(X, IMAGE_SIZE).numpy()
    return X_resized, y

def get_dataset(test_size=0.2):
    X_all, y_all = load_data(TRAIN_IMG_DIR, TRAIN_GT_DIR)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size)
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)
    return (X_train, y_train), (X_test, y_test)
