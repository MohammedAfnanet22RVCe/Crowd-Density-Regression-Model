import tensorflow as tf
from .data_utils import get_dataset
from .config import MODEL_SAVE_PATH

def main():
    _, (X_test, y_test) = get_dataset()
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    loss, mae = model.evaluate(X_test, y_test, verbose=2)
    print(f"📊 Test MSE: {loss:.4f}, MAE: {mae:.4f}")

if __name__ == "__main__":
    main()
