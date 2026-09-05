from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from datetime import datetime
import os


def generate_report(
    disease,
    result,
    confidence,
    original_image_path=None,
    gradcam_image_path=None
):
    # Create reports folder if it doesn't exist
    if not os.path.exists("reports"):
        os.makedirs("reports")

    filename = "reports/Medical_Report.pdf"

    # Create PDF
    pdf = canvas.Canvas(filename, pagesize=letter)
    pdf.setTitle("AI Medical Image Analysis Report")

    width, height = letter

    # =========================================================
    # DETERMINE DISEASE / IMAGE TYPE / SPECIFIC TYPE
    # =========================================================

    result_text = str(result)

    if "Brain" in disease or any(
        tumor_name in result_text
        for tumor_name in ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    ):
        disease_category = "Brain Tumor Detection"
        image_type = "MRI Scan"

        if "Glioma" in result_text:
            tumor_type = "Glioma Tumor"

        elif "Meningioma" in result_text:
            tumor_type = "Meningioma Tumor"

        elif "Pituitary" in result_text:
            tumor_type = "Pituitary Tumor"

        elif "No Tumor" in result_text or "notumor" in result_text.lower():
            tumor_type = "No Tumor Detected"

        else:
            tumor_type = "Brain Tumor Type Not Specified"

    elif "Pneumonia" in disease or "Pneumonia" in result_text:
        disease_category = "Pneumonia Detection"
        image_type = "Chest X-Ray"
        tumor_type = "Not Applicable"

    else:
        disease_category = disease
        image_type = "Medical Image"
        tumor_type = "Not Applicable"

    # =========================================================
    # REPORT HEADER
    # =========================================================

    pdf.setFillColor(colors.HexColor("#0B6E8E"))
    pdf.rect(0, height - 85, width, 85, fill=1, stroke=0)

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 19)

    pdf.drawCentredString(
        width / 2,
        height - 38,
        "MULTI-DISEASE DETECTION SYSTEM"
    )

    pdf.setFont("Helvetica", 10)

    pdf.drawCentredString(
        width / 2,
        height - 58,
        "AI-ASSISTED MEDICAL IMAGE ANALYSIS REPORT"
    )

    # =========================================================
    # REPORT INFORMATION
    # =========================================================

    y = height - 120

    pdf.setFillColor(colors.HexColor("#0B6E8E"))
    pdf.setFont("Helvetica-Bold", 13)

    pdf.drawString(
        60,
        y,
        "REPORT INFORMATION"
    )

    y -= 22

    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 10)

    report_id = datetime.now().strftime(
        "MDDS-%Y%m%d-%H%M%S"
    )

    pdf.drawString(
        60,
        y,
        f"Report ID: {report_id}"
    )

    y -= 18

    generated_date = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    pdf.drawString(
        60,
        y,
        f"Generated Date & Time: {generated_date}"
    )

    # =========================================================
    # EXAMINATION DETAILS
    # =========================================================

    y -= 35

    pdf.setFillColor(colors.HexColor("#0B6E8E"))
    pdf.setFont("Helvetica-Bold", 13)

    pdf.drawString(
        60,
        y,
        "EXAMINATION DETAILS"
    )

    y -= 22

    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 10)

    # Disease Category
    pdf.drawString(
        60,
        y,
        f"Disease Category: {disease_category}"
    )

    y -= 18

    # Specific Type
    pdf.setFont("Helvetica-Bold", 10)

    pdf.drawString(
        60,
        y,
        "Specific Type:"
    )

    pdf.setFont("Helvetica", 10)

    pdf.drawString(
        145,
        y,
        tumor_type
    )

    y -= 18

    # Image Type
    pdf.drawString(
        60,
        y,
        f"Image Type: {image_type}"
    )

    y -= 18

    pdf.drawString(
        60,
        y,
        "Analysis Method: CNN-Based Deep Learning"
    )

    # =========================================================
    # AI ANALYSIS RESULT
    # =========================================================

    y -= 38

    pdf.setFillColor(colors.HexColor("#0B6E8E"))
    pdf.setFont("Helvetica-Bold", 13)

    pdf.drawString(
        60,
        y,
        "AI ANALYSIS RESULT"
    )

    y -= 28

    # Result box
    pdf.setFillColor(colors.HexColor("#F0F8FA"))

    pdf.roundRect(
        55,
        y - 72,
        width - 110,
        82,
        8,
        fill=1,
        stroke=0
    )

    pdf.setFillColor(colors.black)

    # Prediction
    pdf.setFont("Helvetica-Bold", 11)

    pdf.drawString(
        70,
        y - 18,
        "Prediction:"
    )

    pdf.setFont("Helvetica", 11)

    pdf.drawString(
        145,
        y - 18,
        result_text
    )

    # Specific type
    pdf.setFont("Helvetica-Bold", 11)

    pdf.drawString(
        70,
        y - 40,
        "Tumor Type:"
    )

    pdf.setFont("Helvetica", 11)

    pdf.drawString(
        145,
        y - 40,
        tumor_type
    )

    # Confidence
    pdf.setFont("Helvetica-Bold", 11)

    pdf.drawString(
        70,
        y - 62,
        "Confidence:"
    )

    pdf.setFont("Helvetica", 11)

    pdf.drawString(
        145,
        y - 62,
        f"{confidence}%"
    )

    # =========================================================
    # IMAGE ANALYSIS
    # =========================================================

    pdf.showPage()

    # New page header
    pdf.setFillColor(colors.HexColor("#0B6E8E"))
    pdf.rect(
        0,
        height - 65,
        width,
        65,
        fill=1,
        stroke=0
    )

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 17)

    pdf.drawCentredString(
        width / 2,
        height - 38,
        "MEDICAL IMAGE ANALYSIS"
    )

    y = height - 95

    # =========================================================
    # ORIGINAL IMAGE
    # =========================================================

    pdf.setFillColor(colors.HexColor("#0B6E8E"))
    pdf.setFont("Helvetica-Bold", 13)

    pdf.drawString(
        60,
        y,
        "ORIGINAL MEDICAL IMAGE"
    )

    y -= 25

    if (
        original_image_path
        and os.path.exists(original_image_path)
    ):
        try:

            img = ImageReader(original_image_path)

            img_width, img_height = img.getSize()

            max_width = 220
            max_height = 220

            scale = min(
                max_width / img_width,
                max_height / img_height
            )

            display_width = img_width * scale
            display_height = img_height * scale

            x_position = (
                (width / 2)
                - (display_width / 2)
            )

            pdf.drawImage(
                img,
                x_position,
                y - display_height,
                width=display_width,
                height=display_height,
                preserveAspectRatio=True,
                mask="auto"
            )

            y -= display_height + 30

        except Exception as e:

            pdf.setFillColor(colors.black)
            pdf.setFont("Helvetica", 9)

            pdf.drawString(
                60,
                y,
                "Original image could not be embedded."
            )

            print(
                "PDF Original Image Error:",
                e
            )

    else:

        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica", 9)

        pdf.drawString(
            60,
            y,
            "Original image not available."
        )

        y -= 20

    # =========================================================
    # GRAD-CAM IMAGE
    # =========================================================

    y -= 10

    pdf.setFillColor(colors.HexColor("#0B6E8E"))
    pdf.setFont("Helvetica-Bold", 13)

    pdf.drawString(
        60,
        y,
        "GRAD-CAM VISUALIZATION"
    )

    y -= 25

    if (
        gradcam_image_path
        and os.path.exists(gradcam_image_path)
    ):
        try:

            img = ImageReader(
                gradcam_image_path
            )

            img_width, img_height = img.getSize()

            max_width = 220
            max_height = 220

            scale = min(
                max_width / img_width,
                max_height / img_height
            )

            display_width = img_width * scale
            display_height = img_height * scale

            x_position = (
                (width / 2)
                - (display_width / 2)
            )

            pdf.drawImage(
                img,
                x_position,
                y - display_height,
                width=display_width,
                height=display_height,
                preserveAspectRatio=True,
                mask="auto"
            )

            y -= display_height + 25

            # Explanation
            pdf.setFillColor(
                colors.HexColor("#0B6E8E")
            )

            pdf.setFont(
                "Helvetica-Bold",
                11
            )

            pdf.drawCentredString(
                width / 2,
                y,
                "Grad-CAM Color Explanation"
            )

            y -= 18

            pdf.setFillColor(colors.black)
            pdf.setFont("Helvetica", 9)

            explanation_lines = [
                "Red / Orange: Regions receiving stronger attention from the AI model.",
                "Yellow: Regions receiving moderate attention from the AI model.",
                "Green: Regions receiving lower attention from the AI model.",
                "Blue: Regions receiving the least attention from the AI model.",
            ]

            for line in explanation_lines:

                pdf.drawString(
                    75,
                    y,
                    line
                )

                y -= 14

            y -= 5

            pdf.setFillColor(
                colors.HexColor("#555555")
            )

            pdf.setFont(
                "Helvetica-Oblique",
                8
            )

            pdf.drawCentredString(
                width / 2,
                y,
                "Grad-CAM represents model attention and does not indicate"
            )

            y -= 12

            pdf.drawCentredString(
                width / 2,
                y,
                "a confirmed tumor boundary, severity, or exact diagnosis."
            )

        except Exception as e:

            pdf.setFillColor(colors.black)
            pdf.setFont("Helvetica", 9)

            pdf.drawString(
                60,
                y,
                "Grad-CAM image could not be embedded."
            )

            print(
                "PDF Grad-CAM Image Error:",
                e
            )

    else:

        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica", 9)

        pdf.drawString(
            60,
            y,
            "Grad-CAM visualization not available."
        )

    # =========================================================
    # CLINICAL INFORMATION
    # =========================================================

    pdf.showPage()

    pdf.setFillColor(colors.HexColor("#0B6E8E"))
    pdf.rect(
        0,
        height - 65,
        width,
        65,
        fill=1,
        stroke=0
    )

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 17)

    pdf.drawCentredString(
        width / 2,
        height - 38,
        "CLINICAL INFORMATION"
    )

    y = height - 100

    pdf.setFillColor(colors.HexColor("#0B6E8E"))
    pdf.setFont("Helvetica-Bold", 13)

    pdf.drawString(
        60,
        y,
        "CLINICAL INFORMATION"
    )

    y -= 25

    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 10)

    # =========================================================
    # CLINICAL INFORMATION BASED ON TYPE
    # =========================================================

    if "Glioma" in result_text:

        info = (
            "Glioma is a type of tumor that develops from glial "
            "cells in the brain or spinal cord. Further clinical "
            "evaluation and appropriate imaging assessment are "
            "recommended."
        )

        specialist = (
            "Neurologist / Neurosurgeon"
        )

    elif "Meningioma" in result_text:

        info = (
            "Meningioma develops from the membranes surrounding "
            "the brain and spinal cord. Medical evaluation is "
            "recommended to determine its characteristics and "
            "appropriate management."
        )

        specialist = (
            "Neurologist / Neurosurgeon"
        )

    elif "Pituitary" in result_text:

        info = (
            "Pituitary tumors develop in the pituitary gland and "
            "may affect hormone production and other body "
            "functions. Further clinical and hormonal evaluation "
            "may be recommended."
        )

        specialist = (
            "Neurologist / Neurosurgeon / Endocrinologist"
        )

    elif "No Tumor" in result_text:

        info = (
            "No tumor was detected by the AI model in the "
            "uploaded MRI image. If symptoms or clinical "
            "concerns persist, consultation with a qualified "
            "healthcare professional is recommended."
        )

        specialist = (
            "General Physician / Neurologist if clinically indicated"
        )

    elif "Pneumonia" in result_text:

        info = (
            "Pneumonia is an infection that affects the lungs "
            "and may cause symptoms such as cough, fever, chest "
            "discomfort, or breathing difficulty. Clinical "
            "evaluation is recommended."
        )

        specialist = (
            "Pulmonologist / General Physician"
        )

    else:

        info = (
            "The AI model did not identify a specific condition "
            "from the uploaded image. Clinical evaluation should "
            "be considered if symptoms are present."
        )

        specialist = (
            "General Physician"
        )

    # =========================================================
    # DRAW CLINICAL INFORMATION
    # =========================================================

    max_width = 470

    words = info.split()
    line = ""

    for word in words:

        test_line = line + word + " "

        if pdf.stringWidth(
            test_line,
            "Helvetica",
            10
        ) < max_width:

            line = test_line

        else:

            pdf.drawString(
                60,
                y,
                line
            )

            y -= 16
            line = word + " "

    if line:

        pdf.drawString(
            60,
            y,
            line
        )

        y -= 16

    # =========================================================
    # RECOMMENDED SPECIALIST
    # =========================================================

    y -= 18

    pdf.setFillColor(colors.HexColor("#0B6E8E"))
    pdf.setFont("Helvetica-Bold", 13)

    pdf.drawString(
        60,
        y,
        "RECOMMENDED MEDICAL SPECIALIST"
    )

    y -= 25

    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica-Bold", 10)

    pdf.drawString(
        60,
        y,
        specialist
    )

    y -= 20

    pdf.setFont("Helvetica", 9)

    pdf.drawString(
        60,
        y,
        "Specialist recommendation is based only on the AI prediction"
    )

    y -= 14

    pdf.drawString(
        60,
        y,
        "and does not replace professional medical advice."
    )

    # =========================================================
    # DISCLAIMER
    # =========================================================

    y -= 35

    pdf.setFillColor(colors.HexColor("#0B6E8E"))
    pdf.setFont("Helvetica-Bold", 13)

    pdf.drawString(
        60,
        y,
        "IMPORTANT DISCLAIMER"
    )

    y -= 24

    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 9)

    disclaimer = (
        "This report is generated using an AI-based medical image "
        "analysis system and is intended for preliminary screening "
        "and educational purposes only. The prediction should not "
        "be considered a final medical diagnosis. Further clinical "
        "evaluation and confirmation by a qualified healthcare "
        "professional are recommended."
    )

    words = disclaimer.split()
    line = ""

    for word in words:

        test_line = line + word + " "

        if pdf.stringWidth(
            test_line,
            "Helvetica",
            9
        ) < max_width:

            line = test_line

        else:

            pdf.drawString(
                60,
                y,
                line
            )

            y -= 14
            line = word + " "

    if line:

        pdf.drawString(
            60,
            y,
            line
        )

    # =========================================================
    # FOOTER
    # =========================================================

    pdf.setStrokeColor(
        colors.HexColor("#CCCCCC")
    )

    pdf.line(
        60,
        55,
        width - 60,
        55
    )

    pdf.setFillColor(
        colors.HexColor("#666666")
    )

    pdf.setFont(
        "Helvetica",
        8
    )

    pdf.drawCentredString(
        width / 2,
        38,
        "Multi-Disease Detection System | AI-Assisted Medical Screening"
    )

    pdf.drawCentredString(
        width / 2,
        25,
        "For educational and preliminary screening purposes only"
    )

    # =========================================================
    # SAVE PDF
    # =========================================================

    pdf.save()

    print(
        "Medical report generated:",
        filename
    )

    return filename