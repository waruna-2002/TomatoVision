import os
import tensorflow as tf
from src.data_loader import load_datasets, get_data_augmentation
from src.model import build_tomato_classifier
from src.evaluate import evaluate_and_report

def main():
    DATASET_DIR = "dataset"
    MODEL_SAVE_PATH = "models/tomato_model.keras"
    LABELS_PATH = "models/labels.txt"
    
    INITIAL_EPOCHS = 15
    FINE_TUNE_EPOCHS = 10
    INITIAL_LR = 1e-3
    FINE_TUNE_LR = 1e-4

    os.makedirs("models", exist_ok=True)

    print("\n--- Step 1: Loading Dataset ---")
    train_ds, val_ds, class_names = load_datasets(dataset_dir=DATASET_DIR)
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    with open(LABELS_PATH, "w") as f:
        for label in class_names:
            f.write(f"{label}\n")
    print(f"Saved class names to {LABELS_PATH}")

    # Data augmentation pipeline
    data_augmentation = get_data_augmentation()
    train_augmented_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    print("\n--- Step 2: Building Model ---")
    model, base_model = build_tomato_classifier(num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True)
    ]

    print("\n--- Step 3: Phase 1 Training (Feature Extraction) ---")
    model.fit(
        train_augmented_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks
    )

    print("\n--- Step 4: Phase 2 Training (Fine-Tuning) ---")
    base_model.trainable = True
    # Unfreeze only the top layers of MobileNetV2
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
    model.fit(
        train_augmented_ds,
        validation_data=val_ds,
        initial_epoch=INITIAL_EPOCHS,
        epochs=total_epochs,
        callbacks=callbacks
    )

    print("\n--- Step 5: Model Evaluation ---")
    evaluate_and_report(model, val_ds, class_names)
    print(f"\nTraining completed. Saved best model to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()