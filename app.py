from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims
import numpy as np
import os
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('new_model_95.h5')

class_dict = {0: 'Female (Perempuan)', 1: 'Male (Laki-Laki)'}

# GET IMAGE AREA
def get_img_area(path):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    img = cv2.imread(path) # ganti path foto
    # img_resized = cv2.resize(img, (128,128))

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=10)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        img_cropped = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(img_cropped, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) != 0:
            cropped_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cropped_img.jpg')
            cv2.imwrite(cropped_img_path, img_cropped)
            return cropped_img_path

    # cropped_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cropped_img.jpg')
    return path
    # detected_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f'detected_{path}')

    # cv2.imwrite(cropped_img_path, img_cropped)
    # cv2.imwrite(detected_img_path, img_resized)

# PREDICT FUNCTION
def pred_image(path):
    img = load_img(path, target_size=(64,64,3))
    img = img_to_array(img)
    img = img/255.0
    img = img.reshape(1,64,64,3)

    prob = model.predict(img)
    print(prob[0])
    print(prob[0][0])
    if prob[0][0] > 0.5 :
        prob_image = 1
        percentage = round((prob[0][0])*100,1)
    else:
        prob_image = 0
        percentage = round((1 - prob[0][0])*100,1)
    
    return class_dict[prob_image], percentage

# GET REQUEST PREDICTION
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files['image']:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            cropped_path = get_img_area(img_path)
            prediction, percentages = pred_image(cropped_path)
            return render_template('index.html', uploaded_image=image.filename, prediction=prediction, percentage=percentages)
        else:
            return render_template('index.html')

    return render_template('index.html')

# Send Image
@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
