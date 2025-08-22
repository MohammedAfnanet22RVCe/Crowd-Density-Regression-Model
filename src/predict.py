import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from .config import IMAGE_SIZE, MODEL_SAVE_PATH

def predict_image(image_path):
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_array = np.array(img_resized).astype("float32") / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    pred_count = model.predict(input_tensor)[0][0]

    plt.imshow(img)
    plt.title(f"Predicted Crowd Count: {pred_count:.2f}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    predict_image(args.image)
