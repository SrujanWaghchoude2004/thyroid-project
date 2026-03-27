import os
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from predict import extract_full_report, extract_thyroid_values, predict_thyroid, analyze_ultrasound
from PIL import Image, ImageDraw, ImageFont
import io
import textwrap

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

users_db = {}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    msg = ""
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if email in users_db:
            msg = "User already exists. Please login."
        else:
            users_db[email] = password
            return redirect(url_for("login", msg="Signup successful! Please login."))

    return render_template("signup.html", msg=msg)


@app.route("/login", methods=["GET", "POST"])
def login():
    msg = request.args.get("msg", "")

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if email in users_db and users_db[email] == password:
            session["user"] = email
            return redirect(url_for("dashboard"))
        else:
            msg = "Invalid credentials. Try again."

    return render_template("login.html", msg=msg)


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    report_text = ""
    TSH = T3 = T4 = None
    TSH_val = T3_val = T4_val = None
    TSH_status = T3_status = T4_status = None

    diagnosis = None
    image_result = None
    uploaded_image_url = None
    show_ultrasound = False
    risk_score = None
    ai_explanation = None
    recommendation = None
    ultrasound_suggestion = None

    if request.method == "POST":

        # ===== REPORT =====
        report_file = request.files.get("report")
        if report_file:
            report_path = os.path.join(UPLOAD_FOLDER, report_file.filename)
            report_file.save(report_path)

            report_text = extract_full_report(report_path)

            TSH, T3, T4 = extract_thyroid_values(report_text)

            TSH_val = TSH[0] if TSH else None
            T3_val = T3[0] if T3 else None
            T4_val = T4[0] if T4 else None

            def get_status(val_tuple):
                if not val_tuple:
                    return None
                value, low, high = val_tuple
                if value < low:
                    return "Low"
                elif value > high:
                    return "High"
                else:
                    return "Normal"

            TSH_status = get_status(TSH)
            T3_status = get_status(T3)
            T4_status = get_status(T4)

            if TSH and T3 and T4:
                diagnosis = predict_thyroid(TSH, T3, T4)
                if diagnosis in ["Hypothyroidism", "Hyperthyroidism"]:
                    show_ultrasound = True

        # ===== ULTRASOUND =====
        ultrasound_file = request.files.get("ultrasound")
        if ultrasound_file:
            ultrasound_path = os.path.join(UPLOAD_FOLDER, ultrasound_file.filename)
            ultrasound_file.save(ultrasound_path)

            image_result = analyze_ultrasound(ultrasound_path)

            uploaded_image_url = url_for(
                'static',
                filename=f"uploads/{ultrasound_file.filename}"
            )

            if image_result:
                n_type = image_result.get("nodule_type", "").lower()
                shape = image_result.get("shape", "")
                size = image_result.get("size_mm", "")

                if n_type == "benign":
                    ultrasound_suggestion = f"Ultrasound shows a {n_type} nodule ({shape}, {size}). Regular follow-up is recommended."
                elif n_type == "malignant":
                    ultrasound_suggestion = f"ALERT: Ultrasound indicates a {n_type} nodule ({shape}, {size}). Immediate consultation required."
                else:
                    ultrasound_suggestion = f"Ultrasound shows a {n_type} nodule ({shape}, {size}). Consult your doctor."

    # ===== RISK + SUGGESTIONS =====
    if diagnosis:
        if diagnosis == "Normal Thyroid":
            risk_score = 20
            ai_explanation = "Thyroid function appears normal based on lab values."
            recommendation = "Maintain regular checkups."

        elif diagnosis == "Hypothyroidism":
            risk_score = 80
            ai_explanation = "Low thyroid hormone levels detected."
            recommendation = """
Treatment
Levothyroxine (daily, empty stomach)

What to do
- Take medicine same time daily
- Avoid calcium/iron for 30-60 min

Follow-up
- Every 6-8 weeks initially
- Then every 6-12 months

Lifestyle
- Balanced diet
- Exercise regularly
- Manage weight & fatigue
"""

        elif diagnosis == "Hyperthyroidism":
            risk_score = 85
            ai_explanation = "High thyroid hormone levels detected."
            recommendation = """
Treatment options

1. Medicines
Methimazole / Propylthiouracil

2. Symptom control
Propranolol (heart rate, anxiety, tremors)

3. Radioactive iodine therapy

4. Surgery (rare)
- Large goiter
- Cancer suspicion
- Medicine failure
"""

    return render_template(
        "dashboard.html",
        report_text=report_text,
        TSH=TSH_val,
        T3=T3_val,
        T4=T4_val,
        TSH_status=TSH_status,
        T3_status=T3_status,
        T4_status=T4_status,
        diagnosis=diagnosis,
        image_result=image_result,
        show_ultrasound=show_ultrasound,
        uploaded_image_url=uploaded_image_url,
        risk_score=risk_score,
        ai_explanation=ai_explanation,
        recommendation=recommendation,
        ultrasound_suggestion=ultrasound_suggestion
    )


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))


@app.route("/download_report", methods=["POST"])
def download_report():

    TSH = request.form.get("TSH")
    T3 = request.form.get("T3")
    T4 = request.form.get("T4")
    diagnosis = request.form.get("diagnosis")
    risk_score = request.form.get("risk_score")
    symptoms = request.form.get("symptoms")

    image_shape = request.form.get("image_shape")
    image_size = request.form.get("image_size")
    image_area = request.form.get("image_area")
    image_volume = request.form.get("image_volume")
    image_type = request.form.get("image_type")

    img = Image.new("RGB", (900, 1400), "white")
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 42)
        font = ImageFont.truetype("arial.ttf", 24)
        font_bold = ImageFont.truetype("arial.ttf", 26)
        small_font = ImageFont.truetype("arial.ttf", 18)
    except:
        font_title = font = font_bold = small_font = None

    y = 40

    def draw_line():
        nonlocal y
        draw.line((40, y, 860, y), fill="gray", width=2)
        y += 20

    draw.text((200, y), "Thyroid Medical Report", fill="black", font=font_title)
    y += 80
    draw_line()

    draw.text((50, y), "Hormone Values", fill="black", font=font_bold)
    y += 40
    draw.text((50, y), f"TSH: {TSH}", fill="black", font=font); y += 30
    draw.text((50, y), f"T3: {T3}", fill="black", font=font); y += 30
    draw.text((50, y), f"T4: {T4}", fill="black", font=font); y += 40
    draw_line()

    draw.text((50, y), "Diagnosis", fill="black", font=font_bold)
    y += 40
    draw.text((50, y), f"{diagnosis}", fill="red", font=font)
    y += 30
    draw.text((50, y), f"Risk Score: {risk_score}/100", fill="black", font=font)
    y += 40
    draw_line()

    if image_shape:
        draw.text((50, y), "Ultrasound Findings", fill="black", font=font_bold)
        y += 40
        draw.text((50, y), f"Shape: {image_shape}", fill="black", font=font); y += 25
        draw.text((50, y), f"Size: {image_size}", fill="black", font=font); y += 25
        draw.text((50, y), f"Area: {image_area}", fill="black", font=font); y += 25
        draw.text((50, y), f"Volume: {image_volume}", fill="black", font=font); y += 25
        draw.text((50, y), f"Type: {image_type}", fill="black", font=font)
        y += 40
        draw_line()

    if symptoms:
        draw.text((50, y), "Symptoms", fill="black", font=font_bold)
        y += 40
        for line in textwrap.wrap(symptoms, width=60):
            draw.text((50, y), line, fill="black", font=font)
            y += 25
        y += 20
        draw_line()

    draw.text((50, y), "Doctor Recommendation", fill="black", font=font_bold)
    y += 40

    if diagnosis == "Hyperthyroidism":
        rec_lines = [
            "Treatment Options:", "",
            "1. Medicines (First Line)",
            "- Methimazole / Propylthiouracil",
            "- Reduce hormone production", "",
            "2. Symptom Control",
            "- Propranolol",
            "- Controls heart rate, anxiety, tremors", "",
            "3. Radioactive Iodine Therapy",
            "- Reduces thyroid activity", "",
            "4. Surgery (Rare Cases)",
            "- Large goiter",
            "- Cancer suspicion",
            "- Medication failure"
        ]
    elif diagnosis == "Hypothyroidism":
        rec_lines = [
            "Treatment:",
            "- Levothyroxine (daily, empty stomach)", "",
            "What You Should Do:",
            "- Take medicine same time daily",
            "- Avoid calcium/iron for 30-60 minutes", "",
            "Follow-up:",
            "- Blood test every 6-8 weeks initially",
            "- Then every 6-12 months", "",
            "Lifestyle Advice:",
            "- Balanced diet",
            "- Regular exercise",
            "- Manage fatigue & weight"
        ]
    else:
        rec_lines = ["Maintain regular health checkups."]

    for line in rec_lines:
        draw.text((50, y), line, fill="black", font=font)
        y += 25

    disclaimer = "This AI system may make mistakes. Please consult a doctor for medical advice."
    text_width = draw.textlength(disclaimer, font=small_font)
    x_position = (900 - text_width) // 2
    draw.text((x_position, 1350), disclaimer, fill="gray", font=small_font)

    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(
        img_io,
        mimetype='image/jpeg',
        as_attachment=True,
        download_name="thyroid_report.jpg"
    )


if __name__ == "__main__":
    app.run(debug=True)