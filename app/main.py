from flask import Flask
from flask import render_template, request, jsonify
import os
from werkzeug.utils import secure_filename

from .config import Config
from .models.model_loader import ModelLoader
from .models.predictor import DiagnoSpherePredictor

app = Flask(__name__)
app.config.from_object(Config)

# Initialize models at startup
model_loader = ModelLoader()
predictor = DiagnoSpherePredictor(model_loader)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file uploaded',
            'result': None
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'status': 'error', 
            'message': 'No file selected',
            'result': None
        }), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = predictor.predict(filepath)
            if result['status'] == 'error':
                return jsonify(result), 500
            return jsonify(result)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e),
                'result': None
            }), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({
        'status': 'error',
        'message': 'Invalid file type',
        'result': None
    }), 400

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)