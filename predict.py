import cv2
import pytesseract
import numpy as np
import re
import tensorflow as tf
import os
import joblib

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

mlp_data = None
model_path = "models/mlp_model.pkl"

if os.path.exists(model_path):
    mlp_data = joblib.load(model_path)

cnn_model = None
if os.path.exists("models/cnn_model.h5"):
    cnn_model = tf.keras.models.load_model("models/cnn_model.h5")

# Extract full text
def extract_full_report(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)
    text = pytesseract.image_to_string(gray)
    lines = []
    for line in text.split("\n"):
        line = " ".join(line.split())
        if len(line) > 2:
            lines.append(line)
    return "\n".join(lines)

# Extract TSH, T3, T4
import re

def extract_thyroid_values(text):
    def extract_param(param):
        # Matches: TSH 2.5 (0.4 - 4.0)
        pattern = rf"{param}.*?([\d.]+).*?([\d.]+)\s*[-–]\s*([\d.]+)"
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            value = float(match.group(1))
            low = float(match.group(2))
            high = float(match.group(3))
            return value, low, high

        return None, None, None

    TSH = extract_param("TSH")
    T3 = extract_param("T3")
    T4 = extract_param("T4")

    return TSH, T3, T4
# Diagnosis
def predict_thyroid(TSH, T3, T4):
    try:
        tsh_val, tsh_low, tsh_high = TSH
        t3_val, t3_low, t3_high = T3
        t4_val, t4_low, t4_high = T4

        # Status detection
        tsh_status = "low" if tsh_val < tsh_low else "high" if tsh_val > tsh_high else "normal"
        t3_status = "low" if t3_val < t3_low else "high" if t3_val > t3_high else "normal"
        t4_status = "low" if t4_val < t4_low else "high" if t4_val > t4_high else "normal"

        # Diagnosis
        if tsh_status == "low" and (t3_status == "high" or t4_status == "high"):
            return "Hyperthyroidism"

        elif tsh_status == "high" and (t3_status == "low" or t4_status == "low"):
            return "Hypothyroidism"

        else:
            return "Normal Thyroid"

    except:
        return "Could not determine"

# Ultrasound analysis with highlighted problem area
def analyze_ultrasound(path):
    import cv2
    import numpy as np
    import os

    img = cv2.imread(path)
    if img is None:
        return None

    original = img.copy()

    # ===== PREPROCESS =====
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold (better than simple threshold)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # ===== FIND CONTOURS =====
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    nodules = []
    pixel_mm = 0.5  # approximate conversion

    # ===== PROCESS EACH CONTOUR =====
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Remove noise
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        width = w * pixel_mm
        height = h * pixel_mm
        volume = 0.52 * width * height * height / 1000

        shape = "Oval" if abs(w - h) < 20 else "Irregular"

        # ===== RISK CALCULATION =====
        risk = 0

        if area > 5000:
            risk += 2
        if volume > 5:
            risk += 2
        if shape == "Irregular":
            risk += 2

        if risk >= 4:
            risk_level = "High"
            color = (0, 0, 255)  # Red
        elif risk >= 2:
            risk_level = "Medium"
            color = (0, 255, 255)  # Yellow
        else:
            risk_level = "Low"
            color = (0, 255, 0)  # Green

        nodules.append({
            "cnt": cnt,
            "area": area,
            "volume": volume,
            "risk": risk,
            "risk_level": risk_level,
            "color": color,
            "x": x, "y": y, "w": w, "h": h,
            "width": round(width, 2),
            "height": round(height, 2)
        })

    if not nodules:
        return None

    # ===== SORT BY RISK (MAIN NODULE FIRST) =====
    nodules = sorted(nodules, key=lambda x: x["risk"], reverse=True)

    main_nodule = nodules[0]

    # ===== HEATMAP OVERLAY =====
    heatmap = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
    output = cv2.addWeighted(original, 0.7, heatmap, 0.3, 0)

    # ===== DRAW NODULES =====
    for i, n in enumerate(nodules):
        x, y, w, h = n["x"], n["y"], n["w"], n["h"]

        # FIXED BUG: use index instead of comparing dicts
        thickness = 4 if i == 0 else 2

        # Bounding box
        cv2.rectangle(output, (x, y), (x + w, y + h), n["color"], thickness)

        # Center point
        cx = x + w // 2
        cy = y + h // 2
        cv2.circle(output, (cx, cy), 6, n["color"], -1)

        # Label
        label = f"{n['risk_level']} | {n['width']}mm"
        cv2.putText(
            output,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            n["color"],
            2
        )

    # ===== CNN CLASSIFICATION =====
    nodule_type = "Unknown"
    try:
        if cnn_model:
            im = cv2.resize(img, (224, 224)) / 255.0
            im = np.expand_dims(im, 0)
            pred = cnn_model.predict(im)[0][0]
            nodule_type = "Malignant" if pred > 0.5 else "Benign"
    except:
        pass

    # ===== SAVE OUTPUT =====
    output_path = "static/output/advanced_nodule.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, output)

    # ===== RETURN MAIN NODULE DETAILS =====
    return {
        "shape": main_nodule["risk_level"],
        "size_mm": f"{main_nodule['width']} x {main_nodule['height']}",
        "area_px2": int(main_nodule["area"]),
        "volume_cc": round(main_nodule["volume"], 2),
        "nodule_type": nodule_type,
        "image": output_path
    }