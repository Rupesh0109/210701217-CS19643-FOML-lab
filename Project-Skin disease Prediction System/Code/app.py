from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import datetime
import tensorflow as tf
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)

model = tf.keras.models.load_model('./my_model.h5')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    now = datetime.datetime.now()
    unique_filename = now.strftime("%Y%m%d%H%M%S%f") + '_' + secure_filename(file.filename)

    file.save(os.path.join('uploads', unique_filename))

    img_path = os.path.join('uploads', unique_filename)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class_index = np.argmax(predictions)
    class_names = ["Cellulitis","Impetigo","Athlete Foot","Nail Fungus","Ringworm","Cutaneous Larva Migrans","Chickenpox","Shingles"]

    predicted_class = class_names[predicted_class_index]
    confidence_score = 100 * np.max(score)
    confidence_scoref = round(confidence_score, 2)

    os.remove(img_path)

    return jsonify({'DISEASE NAME': predicted_class, 'CONFIDENCE PERCENTAGE': str(confidence_scoref),'INDEX':str(predicted_class_index) })

@app.route('/details')
def show_details():
    disease_name = request.args.get('disease')
    confidence_percentage = request.args.get('confidence')
    index=request.args.get('index')
    with open("data.json", 'r') as file:
        json_data = json.load(file)
        additional_data=json_data[int(index)]

    
    return render_template('details.html', disease=disease_name, confidence=confidence_percentage,data=additional_data)

if __name__ == '__main__':
    app.run(debug=True)
