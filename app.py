import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename

# Initialize Flask
app = Flask(__name__)

# --- Model and Class Configuration ---
try:
    MODEL_PATH = 'model.h5'
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    print("Model output shape:", model.output_shape)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Classes (make sure length matches model output!)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Solutions dictionary
SOLUTIONS = {
    "Apple___Black_rot": {
        "organic": "Prune infected branches and remove mummified fruit. Improve air circulation.",
        "chemical": "Apply fungicides containing Captan or Thiophanate-methyl during the growing season."
    },
    "Corn_(maize)___Common_rust_": {
        "organic": "Plant resistant hybrids. Manage crop residue.",
        "chemical": "Apply foliar fungicides like pyraclostrobin or azoxystrobin if infection is severe."
    },
    "Potato___Late_blight": {
        "organic": "Use certified disease-free seed potatoes. Ensure good drainage and air flow.",
        "chemical": "Apply protectant fungicides like mancozeb or chlorothalonil, especially during cool, moist conditions."
    },
    "default": {
        "organic": "Maintain good plant hygiene and soil health.",
        "chemical": "Consult a local agricultural extension service for specific advice."
    }
}

# --- Helper Function ---
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    return preds

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'})

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, secure_filename(file.filename))
        file.save(file_path)

        # Prediction
        preds = model_predict(file_path, model)
        predicted_class_index = int(np.argmax(preds))
        confidence = float(np.max(preds))

        # ✅ Safe class name selection
        if predicted_class_index < len(CLASS_NAMES):
            predicted_class_name = CLASS_NAMES[predicted_class_index]
        else:
            predicted_class_name = "Unknown_Class"

        solution = SOLUTIONS.get(predicted_class_name, SOLUTIONS["default"])

        os.remove(file_path)

        return jsonify({
            'disease': predicted_class_name.replace('_', ' '),
            'confidence': f"{confidence:.2%}",
            'solution': solution
        })

    return jsonify({'error': 'An unexpected error occurred'})

# --- Main ---
if __name__ == '__main__':
    app.run(debug=True)
