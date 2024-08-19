from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

model = load_model('clasificador_ojos.keras')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_probabilities = predictions[0]  # Obtén las probabilidades para cada clase
    return predicted_class, predicted_probabilities

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            predicted_class, predicted_probabilities = predict_image(file_path)
            class_names = ['Cataratas', 'Glaucoma', 'Retinoplastia_diabetica', 'Sanos']
            predicted_class_name = class_names[predicted_class]
            probabilities_dict = {class_names[i]: float(predicted_probabilities[i]) for i in range(len(class_names))}
            return render_template('result.html', filename=filename, prediction=predicted_class_name, probabilities=probabilities_dict)
    return render_template('index.html')

@app.route('/identificar.html')
def identificar():

    return render_template('identificar.html')

@app.route('/enfermedades.html')
def enfermedades():

    return render_template('enfermedades.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
