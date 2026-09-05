import tensorflow as tf
import numpy as np
import cv2
import os

from tensorflow.keras.models import load_model


# ==============================
# Load Pneumonia Model
# ==============================

model = load_model("models/pneumonia_model.h5")


# ==============================
# Pneumonia Grad-CAM Function
# ==============================

def generate_pneumonia_gradcam(image_path):

    # ==============================
    # Load Original Image
    # ==============================

    original_image = cv2.imread(image_path)

    if original_image is None:

        print("Image not found!")

        return None


    # ==============================
    # Prepare Image
    # ==============================

    img = cv2.resize(
        original_image,
        (128, 128)
    )

    img = img.astype("float32") / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )


    # ==============================
    # Convert to Tensor
    # ==============================

    input_tensor = tf.convert_to_tensor(
        img,
        dtype=tf.float32
    )


    # ==============================
    # Grad-CAM
    # ==============================

    with tf.GradientTape() as tape:

        tape.watch(input_tensor)

        x = input_tensor

        conv_outputs = None

        for layer in model.layers:

            x = layer(x)

            if layer.name == "conv2d_2":

                conv_outputs = x


        predictions = x

        score = predictions[0, 0]


    # ==============================
    # Calculate Gradients
    # ==============================

    grads = tape.gradient(
        score,
        conv_outputs
    )


    if grads is None:

        print("Gradients could not be calculated.")

        return None


    # ==============================
    # Average Gradients
    # ==============================

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )


    # ==============================
    # Remove Batch Dimension
    # ==============================

    conv_outputs = conv_outputs[0]


    # ==============================
    # Create Heatmap
    # ==============================

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )


    # ==============================
    # ReLU
    # ==============================

    heatmap = tf.maximum(
        heatmap,
        0
    )


    # ==============================
    # Normalize Heatmap
    # ==============================

    max_value = tf.reduce_max(
        heatmap
    )


    if float(max_value.numpy()) > 0:

        heatmap = heatmap / max_value


    heatmap = heatmap.numpy()


    # ==============================
    # Prediction
    # ==============================

    score = float(
        predictions[0, 0].numpy()
    )


    if score > 0.5:

        prediction = "Pneumonia"

        confidence = score * 100

    else:

        prediction = "Normal"

        confidence = (1 - score) * 100


    # ==============================
    # Print Result
    # ==============================

    print()
    print("==============================")
    print("Pneumonia Grad-CAM Result")
    print("==============================")


    print(
        "Prediction:",
        prediction
    )


    print(
        "Confidence:",
        round(confidence, 2),
        "%"
    )


    # ==============================
    # Resize Heatmap
    # ==============================

    heatmap = cv2.resize(
        heatmap,
        (
            original_image.shape[1],
            original_image.shape[0]
        )
    )


    # ==============================
    # Convert Heatmap
    # ==============================

    heatmap = np.uint8(
        255 * heatmap
    )


    # ==============================
    # Apply Color Map
    # ==============================

    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    # ==============================
    # Overlay Heatmap
    # ==============================

    result = cv2.addWeighted(
        original_image,
        0.6,
        heatmap_color,
        0.4,
        0
    )


    # ==============================
    # Save Result
    # ==============================

    output_folder = "static/uploads"

    os.makedirs(
        output_folder,
        exist_ok=True
    )


    output_path = os.path.join(
        output_folder,
        "pneumonia_gradcam.jpg"
    )


    success = cv2.imwrite(
        output_path,
        result
    )


    if success:

        print()
        print("Grad-CAM image saved:")
        print(output_path)
        print("==============================")


        return "pneumonia_gradcam.jpg"


    else:

        print()
        print("Failed to save Grad-CAM image.")

        return None