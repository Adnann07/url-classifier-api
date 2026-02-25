from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json, os, re

app = Flask(__name__)
CORS(app)

# ---- Load model & config ----
with open('model_config_final.json') as f:
    config = json.load(f)
with open('char_vocab.json') as f:
    CHAR2IDX = json.load(f)

MAX_LEN       = config['max_len']
LABEL_MAPPING = config['label_mapping']
WHITELIST     = set(config['whitelist'])

interp = tf.lite.Interpreter(model_path='url_classifier_v2.tflite')
interp.allocate_tensors()
inp_d = interp.get_input_details()
out_d = interp.get_output_details()
print("✅ Model loaded")

# ---- Helpers ----
def get_root_domain(url):
    url = re.sub(r'^https?://', '', str(url).strip())
    url = re.sub(r'^www\.', '', url)
    domain = url.split('/')[0].split('?')[0].split('#')[0]
    parts  = domain.split('.')
    return '.'.join(parts[-2:]) if len(parts) >= 2 else domain

def url_to_seq(url):
    seq = [CHAR2IDX.get(c, 1) for c in str(url)[:MAX_LEN]]
    return seq + [0] * (MAX_LEN - len(seq))

def predict_url(url):
    url  = url.strip()
    root = get_root_domain(url)

    if root in WHITELIST:
        return {
            "url":         url,
            "prediction":  "benign",
            "confidence":  99.99,
            "is_safe":     True,
            "whitelisted": True,
            "probabilities": {
                "benign": 99.99, "phishing": 0.0,
                "malware": 0.0,  "defacement": 0.0
            }
        }

    seq  = url_to_seq(url)
    inp  = np.array([seq], dtype=np.int32)
    interp.set_tensor(inp_d[0]['index'], inp)
    interp.invoke()
    probs      = interp.get_tensor(out_d[0]['index'])[0]
    idx        = int(np.argmax(probs))
    label      = LABEL_MAPPING[str(idx)]
    confidence = float(probs[idx]) * 100

    return {
        "url":         url,
        "prediction":  label,
        "confidence":  round(confidence, 2),
        "is_safe":     label == "benign",
        "whitelisted": False,
        "probabilities": {
            LABEL_MAPPING[str(i)]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        }
    }

# ---- Routes ----
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    url  = data.get('url', '').strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    try:
        return jsonify(predict_url(url)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.get_json(force=True)
    urls = data.get('urls', [])
    if not urls or not isinstance(urls, list):
        return jsonify({"error": "Provide a list of URLs"}), 400
    return jsonify({"results": [predict_url(u) for u in urls[:50]]}), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": config['model_version']}), 200

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "URL Classifier API", "version": config['model_version']}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
