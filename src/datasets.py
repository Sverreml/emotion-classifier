from keras.utils import image_dataset_from_directory
from config import BASE_DIR, IMAGE_SIZE, BATCH_SIZE, SEED

def load_datasets():
    train_ds = image_dataset_from_directory(
        BASE_DIR / "train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    val_ds = image_dataset_from_directory(
        BASE_DIR / "validation",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    test_ds = image_dataset_from_directory(
        BASE_DIR / "test",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    return train_ds, val_ds, test_ds
