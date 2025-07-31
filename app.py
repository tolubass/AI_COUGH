from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import torch
import joblib
import io
import base64
import soundfile as sf
from pydub import AudioSegment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import shap
import warnings
import os
import logging
import datetime
from fpdf import FPDF
import uuid

# === Silence warnings and logs ===
warnings.filterwarnings("ignore")
import transformers
transformers.utils.logging.set_verbosity_error()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# === Setup ffmpeg path ===
AudioSegment.converter = r"C:\\Users\\hp\\Desktop\\ffmpeg\\ffmpeg.exe"
os.environ["PATH"] += os.pathsep + r"C:\\Users\\hp\\Desktop\\ffmpeg"

# === Flask app ===
app = Flask(__name__, template_folder='templates', static_folder='static')

# === Load Model & Preprocessors ===
model = joblib.load("tb_rf_model_mini.pkl")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model.eval()

# === Create report folder if not exists ===
os.makedirs("static/reports", exist_ok=True)

# === Functions ===
def load_waveform(uploaded_file):
    audio_bytes = uploaded_file.read()
    ext = uploaded_file.filename.split('.')[-1].lower()
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext)
    wav_io = io.BytesIO()
    audio.set_frame_rate(16000).set_channels(1).export(wav_io, format="wav")
    wav_io.seek(0)
    waveform_np, sr = sf.read(wav_io)
    waveform = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)
    return waveform, sr

def generate_spectrogram_base64(waveform):
    buf = io.BytesIO()
    plt.figure(figsize=(8, 4))
    plt.specgram(waveform.squeeze().numpy(), Fs=16000, NFFT=1024, noverlap=512, cmap='inferno')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_shap_image(input_vector):
    try:
        explainer = shap.Explainer(model.predict_proba, feature_names=[f"F{i}" for i in range(len(input_vector))])
        shap_values = explainer([input_vector])
        buf = io.BytesIO()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception:
        return None

def generate_pdf_report(result, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="TB Risk Prediction Report", ln=True, align="C")
    pdf.ln(10)
    for key, value in result.items():
        if key not in ['spectrogram_base64', 'shap_base64', 'report_link', 'recommendation', 'disclaimer']:
            pdf.cell(200, 10, txt=f"{key.capitalize()}: {value}", ln=True)

    img_path = f"static/reports/{filename}_spectrogram.png"
    if result.get('spectrogram_base64'):
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(result['spectrogram_base64']))
        pdf.image(img_path, w=160)
        os.remove(img_path)

    pdf.ln(5)
    if result.get('recommendation'):
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(190, 10, txt=f"Recommendation: {result['recommendation']}")

    if result.get('disclaimer'):
        pdf.set_font("Arial", size=10, style='I')
        pdf.multi_cell(190, 10, txt=f"Disclaimer: {result['disclaimer']}")

    if result.get('shap_base64'):
        shap_path = f"static/reports/{filename}_shap.png"
        with open(shap_path, "wb") as f:
            f.write(base64.b64decode(result['shap_base64']))
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Feature Contributions (SHAP)", ln=True)
        pdf.image(shap_path, w=160)
        os.remove(shap_path)

    report_path = f"static/reports/{filename}.pdf"
    pdf.output(report_path, "F")
    return report_path

# === Routes ===
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/faq')
def faq():
    return render_template("faq.html")

@app.route('/predict', methods=["GET"])
def predict_form():
    return render_template("prediction.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        age = int(request.form['age'])
        cough_days = int(request.form['reported_cough_dur'])
        hemoptysis = int(request.form['hemoptysis'])
        temperature = float(request.form['temperature'])
        weight_loss = int(request.form['weight_loss'])
        heart_rate = float(request.form['heart_rate'])

        audio_file = request.files.get('audio_file')
        embeddings = np.zeros(768)
        spectrogram_base64 = None

        if audio_file and audio_file.filename:
            waveform, sr = load_waveform(audio_file)
            inputs = processor(waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = wav2vec_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            spectrogram_base64 = generate_spectrogram_base64(waveform)

        clinical_features = [age, cough_days, hemoptysis, temperature, weight_loss, heart_rate]
        input_vector = np.concatenate([clinical_features, embeddings])

        proba = model.predict_proba([input_vector])[0][1]
        prediction = 1 if proba >= 0.55 else 0
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if prediction == 1:
            recommendation = (
                f"Based on the input details, including a cough duration of {cough_days} days, "
                f"{'presence of hemoptysis (coughing up blood)' if hemoptysis else 'no hemoptysis reported'}, "
                f"a body temperature of {temperature} C, "
                f"{'observed weight loss' if weight_loss else 'no significant weight loss'}, "
                f"and a heart rate of {heart_rate} bpm, the analysis indicates a likelihood of active tuberculosis.\n\n"
                "It is strongly recommended that you seek prompt medical evaluation at a recognized health facility or tuberculosis treatment center. "
                "Diagnostic assessments such as chest radiography, sputum examination, or GeneXpert testing may be necessary for confirmation. "
                "Timely diagnosis and initiation of therapy are critical to preventing complications and limiting disease transmission."
            )
        else:
            recommendation = (
                f"Based on your provided information, including a cough lasting {cough_days} days, "
                f"{'presence of hemoptysis' if hemoptysis else 'no signs of hemoptysis'}, "
                f"body temperature of {temperature} C, "
                f"{'signs of weight loss' if weight_loss else 'no weight loss'}, "
                f"and a heart rate of {heart_rate} bpm, the symptoms do not strongly suggest active tuberculosis at this time.\n\n"
                "However, continued monitoring of your health is advised. If symptoms worsen or additional signs such as persistent fatigue, chest pain, "
                "night sweats, or difficulty breathing develop, please consult a healthcare provider promptly.\n\n"
                "Support your recovery with proper rest, hydration, and nutrition, and consider scheduling a follow-up consultation to explore other potential causes."
            )

        result = {
            "date": now,
            "prediction": "Probable TB" if prediction else "Unlikely TB",
            "confidence": f"{proba * 100:.2f}%",
            "recommendation": recommendation,
            "disclaimer": "This is an AI-based estimate. Always consult a healthcare provider."
        }

        if spectrogram_base64:
            result["spectrogram_base64"] = spectrogram_base64

        shap_base64 = generate_shap_image(input_vector)
        if shap_base64:
            result["shap_base64"] = shap_base64

        unique_id = str(uuid.uuid4())[:8]
        report_path = generate_pdf_report(result, unique_id)
        result["report_link"] = f"/static/reports/{unique_id}.pdf"

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Prediction Error: {str(e)}"})

@app.route('/static/reports/<filename>')
def download_report(filename):
    return send_from_directory("static/reports", filename)

# === Main ===
if __name__ == '__main__':
    app.run(debug=True)
