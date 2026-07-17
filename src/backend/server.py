from flask import Flask, request, jsonify, render_template, Response
import numpy as np
import cv2
import base64
import os
import json
import requests as req_lib
from improved_pest_detection import PestDetectionSystem

app = Flask(__name__)

ESP32_STREAM_URL = "http://10.22.71.19:80/stream"

# Create required folders
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists('templates'):
    os.makedirs('templates')

# FIX 2b: Load class names from JSON so labels match the trained model
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# FIX 1: Corrected model filename from tinylitenetfinal.keras → tinylitenet_best.keras
system = PestDetectionSystem(
    model_path='models/tinylitenet_best.keras',
    class_names=class_names
)
system.compile_model()
print("✅ Model loaded!")

# Global result store
last_result = {
    'disease': 'Waiting...',
    'confidence': 0,
    'affected_area': 0,
    'feature_type': '',
    'led': 'GREEN',
    'imagepath': None,
    'affected_image': None
}

# FIX 1: ESP_IP must be a STRING (was 0.0.0.0 without quotes = syntax crash)
ESP_IP = "10.22.71.19"  # ← PLACEHOLDER: Update when needed

def adjust_confidence(disease, confidence, humidity, leaf_moisture):
    """
    Adjusts CNN confidence based on humidity + leaf moisture sensor readings.
    Each disease group reacts differently to wet or dry conditions.

    FUNGAL  → thrive in HIGH humidity + wet leaves
    BACTERIAL → thrive in MODERATE-HIGH humidity + wet leaves
    VIRAL   → NOT affected by humidity (spread by insects, not moisture)
    PEST (spider mites) → thrive in DRY + LOW humidity conditions
    HEALTHY → no adjustment
    """

    # FUNGAL: high humidity + wet leaf = more likely → boost confidence
    fungal_diseases = [
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Tomato_Early_blight',
        'Tomato_Late_blight',
        'Tomato_Leaf_Mold',
        'Tomato_Septoria_leaf_spot',
        'Tomato__Target_Spot'
    ]

    # BACTERIAL: wet conditions also favour bacterial spread
    bacterial_diseases = [
        'Pepper__bell___Bacterial_spot',
        'Tomato_Bacterial_spot'
    ]

    # PEST: dry conditions favour spider mites
    dry_condition_pests = [
        'Tomato_Spider_mites_Two_spotted_spider_mite'
    ]

    # VIRAL: no adjustment — not moisture dependent
    # Tomato__Tomato_YellowLeaf__Curl_Virus
    # Tomato__Tomato_mosaic_virus

    if disease in fungal_diseases:
        if humidity > 75 and leaf_moisture > 70:
            confidence += 7    # very wet = strongly favours fungal
        elif humidity > 60 and leaf_moisture > 50:
            confidence += 3    # moderately wet = slightly favours fungal
        elif humidity < 40 and leaf_moisture < 30:
            confidence -= 7    # dry = unlikely to be fungal

    elif disease in bacterial_diseases:
        if humidity > 70 and leaf_moisture > 65:
            confidence += 5    # wet leaves spread bacteria
        elif humidity < 35 and leaf_moisture < 25:
            confidence -= 5    # very dry = less likely bacterial

    elif disease in dry_condition_pests:
        if humidity < 40 and leaf_moisture < 30:
            confidence += 6    # dry = spider mites very likely
        elif humidity > 75 and leaf_moisture > 70:
            confidence -= 6    # wet = spider mites unlikely

    return max(0, min(confidence, 100))   # clamp to 0-100

@app.route('/predict', methods=['POST'])
def predict():
    global last_result
    try:
        data = request.json

        # Decode base64 image from ESP32
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # FIX 2: Check image decoded properly
        if image is None:
            return jsonify({'error': 'Image decode failed. Check ESP32 base64 encoding.'})

        # Save original image for dashboard
        imagepath = '/static/latest.jpg'
        cv2.imwrite('static/latest.jpg', image)

        # Read sensor values safely with defaults
        humidity = float(data.get('humidity', 50))
        leaf_moisture = float(data.get('leaf_moisture', 50))

        # CNN prediction
        results = system.predict_single_image_from_array(image)
        disease = results['predicted_class']
        confidence = results['confidence']
        confidence = adjust_confidence(disease, confidence, humidity, leaf_moisture)

        # FIX 3: Safely check if segmenter exists before calling it
        if hasattr(system, 'segmenter') and system.segmenter is not None:
            segmentation = system.segmenter.segment_image(image)
            mask = segmentation['binary']
            affected_area = np.sum(mask == 0) / mask.size * 100
            affected_img = segmentation['masked']
            affected_path = '/static/affected_area.jpg'
            cv2.imwrite('static/affected_area.jpg', affected_img)
        else:
            print("⚠️ segmenter not found — skipping segmentation")
            affected_area = 0
            affected_path = None

        led = 'GREEN' if disease == 'Healthy' else 'RED'
        feature_description = (
            "Significant tissue damage visible in segmented regions."
            if disease != 'Healthy'
            else "Leaf structure appears normal."
        )

        last_result = {
            'disease': disease,
            'confidence': round(confidence, 2),
            'affected_area': round(affected_area, 1),
            'feature_type': feature_description,
            'led': led,
            'imagepath': imagepath,
            'affected_image': affected_path
        }

        return jsonify(last_result)

    except KeyError as e:
        return jsonify({'error': f'Missing field in ESP32 data: {str(e)}'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/stream')
def stream_proxy():
    """Proxy ESP32 stream through Flask to avoid browser CORS block"""
    try:
        esp32 = req_lib.get(ESP32_STREAM_URL, stream=True, timeout=10)
        return Response(
            esp32.iter_content(chunk_size=1024),
            content_type=esp32.headers.get('content-type', 'multipart/x-mixed-replace')
        )
    except Exception as e:
        return jsonify({'error': f'ESP32 stream unavailable: {str(e)}'}), 503

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', result=last_result)

# NEW: Dashboard polls this to get latest result as JSON
@app.route('/api/result')
def api_result():
    return jsonify(last_result)

# FIX 4: Added /status route for quick server health check
@app.route('/status')
def status():
    return jsonify({'server': 'running', 'last_disease': last_result['disease']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)