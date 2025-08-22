from .data_utils import get_dataset
from .model import build_model
from .config import EPOCHS, BATCH_SIZE, VAL_SPLIT, MODEL_SAVE_PATH

def main():
    (X_train, y_train), _ = get_dataset()

    model = build_model()
    model.summary()

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT
    )

    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
