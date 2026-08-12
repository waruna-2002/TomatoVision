import os
import tensorflow as tf

KERAS_MODEL_PATH = "models/tomato_model.keras"
TFLITE_MODEL_PATH = "models/model.tflite"

def convert_to_tflite():
    if not os.path.exists(KERAS_MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at '{KERAS_MODEL_PATH}'. Run 'train.py' first.")

    print(f"Loading trained model from {KERAS_MODEL_PATH}...")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)

    print("Converting model to TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print("=" * 50)
    print(f"Conversion successful!")
    print(f"TFLite model saved to : {TFLITE_MODEL_PATH}")
    print(f"Model file size       : {size_mb:.2f} MB")
    print("=" * 50)

if __name__ == "__main__":
    convert_to_tflite()