import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# Cargar el modelo
model = load_model('clasificador_ojos.keras')

# Cargar y preprocesar la imagen
def load_and_preprocess_image(img_path, img_height=180, img_width=180):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalización
    return img_array

# Hacer predicción
def predict_image_class(img_path, model):
    img_array = load_and_preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class, prediction

# Mapear índices a nombres de clases
class_labels = ['Cataratas', 'Glaucoma', 'Retinoplastia_diabetica', 'Sanos']

def get_class_label(predicted_class, class_labels):
    return class_labels[predicted_class[0]]

# Leer el nombre de la imagen desde la consola
img_name = input("Introduce el nombre del archivo de la imagen (por ejemplo, imagen1.jpg): ")
img_path = os.path.join('data-test', img_name)  # Ajustar al directorio 'imagen_test'

# Verifica que la imagen exista
if not os.path.exists(img_path):
    print(f"Error: la imagen '{img_path}' no existe.")
else:
    # Probar con la imagen proporcionada
    predicted_class, prediction = predict_image_class(img_path, model)
    class_label = get_class_label(predicted_class, class_labels)

    print(f'Clase predecida: {class_label}')
    print(f'Class Probabilities: {prediction}')

    # Visualizar la imagen con la predicción
    def show_image_with_prediction(img_path, class_label):
        img = image.load_img(img_path)
        plt.imshow(img)
        plt.title(f'El Ojo esta : {class_label}')
        plt.axis('off')
        plt.show()

    show_image_with_prediction(img_path, class_label)
