import tensorflow as tf
from tensorflow.keras import layers

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def get_data_augmentation():
    """Defines image augmentation pipeline for training."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")

def load_datasets(dataset_dir="dataset", img_size=IMG_SIZE, batch_size=BATCH_SIZE, validation_split=0.2, seed=42):
    """
    Loads training and validation datasets directly from folder structure
    using stratified random split.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    class_names = train_ds.class_names

    # Configure dataset for performance (caching and prefetching)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names