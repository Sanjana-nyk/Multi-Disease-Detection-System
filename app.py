from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import os

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    redirect,
    url_for,
    session
)

from werkzeug.utils import secure_filename

from report_generator import generate_report


app = Flask(__name__)


# ============================================================
# LOGIN / SESSION CONFIGURATION
# ============================================================

app.secret_key = "multi_disease_detection_secret_key"


# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# LOGIN CREDENTIALS
# ============================================================

LOGIN_USERNAME = "admin"
LOGIN_PASSWORD = "1234"


# ============================================================
# LOAD MODELS
# ============================================================

brain_tumor_model = load_model(
    "models/brain_tumor_model.h5"
)

pneumonia_model = load_model(
    "models/pneumonia_model.h5"
)


# ============================================================
# GRAD-CAM HELPER
# ============================================================

def forward_with_gradcam(model, input_tensor):

    last_conv_output = None

    def process_layers(layers, x):

        nonlocal last_conv_output

        for layer in layers:

            # Nested Sequential / Model
            if isinstance(
                layer,
                (
                    tf.keras.models.Sequential,
                    tf.keras.models.Model
                )
            ):

                x = process_layers(
                    layer.layers,
                    x
                )

            else:

                x = layer(
                    x,
                    training=False
                )

                # Capture Conv2D output
                if isinstance(
                    layer,
                    tf.keras.layers.Conv2D
                ):

                    last_conv_output = x

        return x

    predictions = process_layers(
        model.layers,
        input_tensor
    )

    return last_conv_output, predictions


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    image_path,
    output_filename
):

    try:

        # ----------------------------------------------------
        # READ ORIGINAL IMAGE
        # ----------------------------------------------------

        original_image = cv2.imread(
            image_path
        )

        if original_image is None:

            print(
                "Grad-CAM Error: Unable to read image."
            )

            return None

        # ----------------------------------------------------
        # PREPARE IMAGE
        # ----------------------------------------------------

        input_image = cv2.resize(
            original_image,
            (128, 128)
        )

        input_image = (
            input_image.astype("float32")
            / 255.0
        )

        input_image = np.expand_dims(
            input_image,
            axis=0
        )

        input_tensor = tf.convert_to_tensor(
            input_image,
            dtype=tf.float32
        )

        # ----------------------------------------------------
        # CALCULATE GRADIENTS
        # ----------------------------------------------------

        with tf.GradientTape() as tape:

            tape.watch(input_tensor)

            conv_outputs, predictions = (
                forward_with_gradcam(
                    model,
                    input_tensor
                )
            )

            if conv_outputs is None:

                print(
                    "Grad-CAM Error: "
                    "No Conv2D activation was captured."
                )

                return None

            # ------------------------------------------------
            # DETERMINE CLASS OUTPUT
            # ------------------------------------------------

            if len(predictions.shape) == 2:

                number_of_classes = (
                    predictions.shape[-1]
                )

                # Binary classification
                if number_of_classes == 1:

                    class_output = predictions[:, 0]

                # Multi-class classification
                else:

                    predicted_index = tf.argmax(
                        predictions[0]
                    )

                    class_output = predictions[
                        :,
                        predicted_index
                    ]

            else:

                class_output = predictions

        # ----------------------------------------------------
        # GRADIENTS
        # ----------------------------------------------------

        grads = tape.gradient(
            class_output,
            conv_outputs
        )

        if grads is None:

            print(
                "Grad-CAM Error: Gradients are None."
            )

            return None

        # ----------------------------------------------------
        # GLOBAL AVERAGE POOLING
        # ----------------------------------------------------

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        # ----------------------------------------------------
        # REMOVE BATCH DIMENSION
        # ----------------------------------------------------

        conv_outputs = conv_outputs[0]

        # ----------------------------------------------------
        # CREATE HEATMAP
        # ----------------------------------------------------

        heatmap = tf.reduce_sum(
            conv_outputs * pooled_grads,
            axis=-1
        )

        # ----------------------------------------------------
        # RELU
        # ----------------------------------------------------

        heatmap = tf.maximum(
            heatmap,
            0
        )

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        max_value = tf.reduce_max(
            heatmap
        )

        if float(max_value) > 0:

            heatmap = (
                heatmap / max_value
            )

        heatmap = heatmap.numpy()

        # ----------------------------------------------------
        # RESIZE HEATMAP
        # ----------------------------------------------------

        heatmap = cv2.resize(
            heatmap,
            (
                original_image.shape[1],
                original_image.shape[0]
            )
        )

        # ----------------------------------------------------
        # CONVERT TO 8-BIT
        # ----------------------------------------------------

        heatmap = np.uint8(
            255 * heatmap
        )

        # ----------------------------------------------------
        # APPLY COLOR MAP
        # ----------------------------------------------------

        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        # ----------------------------------------------------
        # OVERLAY HEATMAP
        # ----------------------------------------------------

        gradcam_result = cv2.addWeighted(
            original_image,
            0.60,
            heatmap_color,
            0.40,
            0
        )

        # ----------------------------------------------------
        # SAVE GRAD-CAM IMAGE
        # ----------------------------------------------------

        gradcam_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            output_filename
        )

        success = cv2.imwrite(
            gradcam_path,
            gradcam_result
        )

        if not success:

            print(
                "Grad-CAM Error: "
                "Unable to save Grad-CAM image."
            )

            return None

        print(
            "Grad-CAM generated:",
            gradcam_path
        )

        return output_filename

    except Exception as e:

        print(
            "Grad-CAM Error:",
            e
        )

        return None


# ============================================================
# LOGIN PAGE
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    # --------------------------------------------------------
    # IF ALREADY LOGGED IN
    # --------------------------------------------------------

    if session.get("logged_in"):

        return redirect(
            url_for("home")
        )

    # --------------------------------------------------------
    # LOGIN FORM
    # --------------------------------------------------------

    if request.method == "POST":

        username = request.form.get(
            "username"
        )

        password = request.form.get(
            "password"
        )

        # ----------------------------------------------------
        # CHECK LOGIN DETAILS
        # ----------------------------------------------------

        if (
            username == LOGIN_USERNAME
            and password == LOGIN_PASSWORD
        ):

            session["logged_in"] = True
            session["username"] = username

            return redirect(
                url_for("home")
            )

        else:

            return render_template(
                "login.html",
                error="Invalid username or password."
            )

    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    # --------------------------------------------------------
    # LOGIN PROTECTION
    # --------------------------------------------------------

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )

    return render_template(
        "index.html"
    )


# ============================================================
# BRAIN TUMOR PAGE
# ============================================================

@app.route("/brain_tumor")
def brain_tumor():

    # --------------------------------------------------------
    # LOGIN PROTECTION
    # --------------------------------------------------------

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )

    return render_template(
        "brain_tumor.html"
    )


# ============================================================
# PNEUMONIA PAGE
# ============================================================

@app.route("/pneumonia")
def pneumonia():

    # --------------------------------------------------------
    # LOGIN PROTECTION
    # --------------------------------------------------------

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )

    return render_template(
        "pneumonia.html"
    )


# ============================================================
# BRAIN TUMOR PREDICTION
# ============================================================

@app.route(
    "/predict_brain_tumor",
    methods=["POST"]
)
def predict_brain_tumor():

    # --------------------------------------------------------
    # LOGIN PROTECTION
    # --------------------------------------------------------

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if "image" not in request.files:

        return "No image uploaded.", 400

    image = request.files["image"]

    if image.filename == "":

        return "No image selected.", 400

    # --------------------------------------------------------
    # SECURE FILE NAME
    # --------------------------------------------------------

    filename = secure_filename(
        image.filename
    )

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    image.save(
        image_path
    )

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    img = cv2.imread(
        image_path
    )

    if img is None:

        return "Invalid image file.", 400

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    img = cv2.resize(
        img,
        (128, 128)
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    img = img.astype(
        "float32"
    ) / 255.0

    # --------------------------------------------------------
    # ADD BATCH DIMENSION
    # --------------------------------------------------------

    img = np.expand_dims(
        img,
        axis=0
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = brain_tumor_model.predict(
        img,
        verbose=0
    )

    predicted_class = np.argmax(
        prediction[0]
    )

    confidence = round(
        float(
            np.max(prediction[0])
        ) * 100,
        2
    )

    probabilities = (
        prediction[0] * 100
    )

    # --------------------------------------------------------
    # CLASS NAMES
    # --------------------------------------------------------

    classes = [
        "glioma",
        "meningioma",
        "notumor",
        "pituitary"
    ]

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if classes[predicted_class] == "glioma":

        result = "Glioma Tumor Detected"

        disease_info = (
            "Glioma is a tumor that develops from "
            "glial cells in the brain. Further medical "
            "evaluation is recommended."
        )

    elif classes[predicted_class] == "meningioma":

        result = "Meningioma Tumor Detected"

        disease_info = (
            "Meningioma usually develops in the membranes "
            "surrounding the brain and is often slow-growing."
        )

    elif classes[predicted_class] == "pituitary":

        result = "Pituitary Tumor Detected"

        disease_info = (
            "Pituitary tumors develop in the pituitary gland "
            "and may affect hormone production."
        )

    else:

        result = "No Tumor Detected"

        disease_info = (
            "No tumor was detected in the uploaded MRI image. "
            "This AI prediction should not replace "
            "professional diagnosis."
        )

    # --------------------------------------------------------
    # PREDICTION BREAKDOWN
    # --------------------------------------------------------

    prediction_details = {

        "Glioma": round(
            float(probabilities[0]),
            2
        ),

        "Meningioma": round(
            float(probabilities[1]),
            2
        ),

        "No Tumor": round(
            float(probabilities[2]),
            2
        ),

        "Pituitary": round(
            float(probabilities[3]),
            2
        )

    }

    # ========================================================
    # GENERATE GRAD-CAM FIRST
    # ========================================================

    gradcam_file = generate_gradcam(
        brain_tumor_model,
        image_path,
        "brain_gradcam.jpg"
    )

    # ========================================================
    # GENERATE MEDICAL REPORT
    # ========================================================

    gradcam_path = None

    if gradcam_file:

        gradcam_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            gradcam_file
        )

    generate_report(
        "Brain Tumor Detection",
        result,
        confidence,
        original_image_path=image_path,
        gradcam_image_path=gradcam_path
    )

    # --------------------------------------------------------
    # RESULT PAGE
    # --------------------------------------------------------

    return render_template(

        "result.html",

        disease="Brain Tumor Detection",

        result=result,

        confidence=confidence,

        image_file=filename,

        prediction_details=prediction_details,

        disease_info=disease_info,

        report_available=True,

        gradcam_file=gradcam_file

    )


# ============================================================
# PNEUMONIA PREDICTION
# ============================================================

@app.route(
    "/predict_pneumonia",
    methods=["POST"]
)
def predict_pneumonia():

    # --------------------------------------------------------
    # LOGIN PROTECTION
    # --------------------------------------------------------

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if "image" not in request.files:

        return "No image uploaded.", 400

    image = request.files["image"]

    if image.filename == "":

        return "No image selected.", 400

    # --------------------------------------------------------
    # SECURE FILE NAME
    # --------------------------------------------------------

    filename = secure_filename(
        image.filename
    )

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    image.save(
        image_path
    )

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    img = cv2.imread(
        image_path
    )

    if img is None:

        return "Invalid image file.", 400

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    img = cv2.resize(
        img,
        (128, 128)
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    img = img.astype(
        "float32"
    ) / 255.0

    # --------------------------------------------------------
    # ADD BATCH DIMENSION
    # --------------------------------------------------------

    img = np.expand_dims(
        img,
        axis=0
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = pneumonia_model.predict(
        img,
        verbose=0
    )

    score = float(
        prediction[0][0]
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if score > 0.5:

        result = "Pneumonia Detected"

        confidence = round(
            score * 100,
            2
        )

        disease_info = (
            "Pneumonia is an infection that affects "
            "the lungs and may cause cough, fever, "
            "and breathing difficulty."
        )

    else:

        result = "Normal"

        confidence = round(
            (1 - score) * 100,
            2
        )

        disease_info = (
            "No signs of pneumonia were detected "
            "in the uploaded X-ray image."
        )

    # ========================================================
    # GENERATE GRAD-CAM FIRST
    # ========================================================

    gradcam_file = generate_gradcam(
        pneumonia_model,
        image_path,
        "pneumonia_gradcam_result.jpg"
    )

    # ========================================================
    # GENERATE MEDICAL REPORT
    # ========================================================

    gradcam_path = None

    if gradcam_file:

        gradcam_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            gradcam_file
        )

    generate_report(
        "Pneumonia Detection",
        result,
        confidence,
        original_image_path=image_path,
        gradcam_image_path=gradcam_path
    )

    # --------------------------------------------------------
    # RESULT PAGE
    # --------------------------------------------------------

    return render_template(

        "result.html",

        disease="Pneumonia Detection",

        result=result,

        confidence=confidence,

        image_file=filename,

        prediction_details=None,

        disease_info=disease_info,

        report_available=True,

        gradcam_file=gradcam_file

    )


# ============================================================
# DOWNLOAD MEDICAL REPORT
# ============================================================

@app.route(
    "/download_report"
)
def download_report():

    # --------------------------------------------------------
    # LOGIN PROTECTION
    # --------------------------------------------------------

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )

    report_path = os.path.join(
        "reports",
        "Medical_Report.pdf"
    )

    if not os.path.exists(
        report_path
    ):

        return (
            "Medical report not found.",
            404
        )

    return send_file(

        report_path,

        as_attachment=True,

        download_name="Medical_Report.pdf"

    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )