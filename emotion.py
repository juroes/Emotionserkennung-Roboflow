import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog

model = tf.keras.models.load_model('emotions_model.h5')

class_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisieren

    # Vorhersage
    predictions = model.predict(img_array)
    
    # Bestimme Klasse mit höchster wslkt
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_index]
    predicted_probability = predictions[0][predicted_class_index]

    return predicted_class, predicted_probability

def upload_image():
    file_path = filedialog.askopenfilename(title="Wähle ein Bild aus", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        class_name, probability = predict_image(file_path)
        result_label.config(text=f"Vorhergesagte Klasse: {class_name}\nWahrscheinlichkeit: {probability*100:.2f}%")

# GUI mit Tkinter
root = tk.Tk()
root.title("Emotions-Klassifikation")

upload_button = tk.Button(root, text="Bild hochladen", command=upload_image)
upload_button.pack(pady=20)

result_label = tk.Label(root, text="Wähle ein Bild aus, um die Vorhersage zu sehen", font=("Arial", 14))
result_label.pack(pady=20)

root.mainloop()