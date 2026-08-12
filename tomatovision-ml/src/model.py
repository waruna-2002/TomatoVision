import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def build_tomato_classifier(num_classes=4, input_shape=(224, 224, 3)):
    """
    Builds a transfer learning model based on MobileNetV2 optimized for edge deployment.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # Preprocess inputs for MobileNetV2 (scales pixels to [-1, 1])
    x = preprocess_input(inputs)
    
    # Feature extraction
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="TomatoVision_MobileNetV2")
    return model, base_model