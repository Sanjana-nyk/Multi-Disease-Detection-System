import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.models import load_model


# ==============================
# Load Brain Tumor Model
# ==============================

model = load_model("models/brain_tumor_model.h5")


# ==============================
# Class Names
# ==============================

classes = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# ==============================
# Image Path
# ==============================

image_path = "static/uploads/Te-gl_9.jpg"


# ==============================
# Load Image
# ==============================

original_image = cv2.imread(image_path)

if original_image is None:
    print("Image not found!")
    exit()


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
# Create Input Tensor
# ==============================

input_tensor = tf.convert_to_tensor(
    img,
    dtype=tf.float32
)


# ==============================
# Grad-CAM
# ==============================

last_conv_layer = model.get_layer(
    "conv2d_5"
)


with tf.GradientTape() as tape:

    tape.watch(input_tensor)

    x = input_tensor

    conv_outputs = None

    for layer in model.layers:

        x = layer(x)

        if layer.name == "conv2d_5":

            conv_outputs = x

    predictions = x

    predicted_class = tf.argmax(
        predictions[0]
    )

    class_output = predictions[
        0,
        predicted_class
    ]


# ==============================
# Calculate Gradients
# ==============================

grads = tape.gradient(
    class_output,
    conv_outputs
)


if grads is None:

    print("Gradients could not be calculated.")

    exit()


# ==============================
# Average Gradients
# ==============================

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0, 1, 2)
)


# ==============================
# Create Heatmap
# ==============================

conv_outputs = conv_outputs[0]

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
# Normalize
# ==============================

max_value = tf.reduce_max(
    heatmap
)

if max_value > 0:

    heatmap = heatmap / max_value


heatmap = heatmap.numpy()


# ==============================
# Prediction
# ==============================

predicted_class = int(
    predicted_class.numpy()
)

confidence = float(
    predictions[0, predicted_class].numpy()
) * 100


# ==============================
# Print Result
# ==============================

print()
print("==============================")
print("Grad-CAM Result")
print("==============================")

print(
    "Prediction:",
    classes[predicted_class]
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

output_path = "static/uploads/gradcam_result.jpg"


cv2.imwrite(
    output_path,
    result
)


print()
print("Grad-CAM image saved:")
print(output_path)

print("==============================")