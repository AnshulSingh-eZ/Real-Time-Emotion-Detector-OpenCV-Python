import cv2
import keras.models
from keras.models import load_model
import numpy as np



model = load_model('Emotion_Detector_Model.h5')  
har_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(har_file)

def extract(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

webcam = cv2.VideoCapture(0)
labels = {0:'angry', 1:'disgust', 2:'fear', 3: 'happy', 4 : 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    i,im = webcam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im,1.3,5)
    try:
        for p,q,r,s in faces:
            image = gray[q:q+s, p:p+s]
            cv2.rectangle(im, (p,q), (p+r, q+s), (0, 255, 0), 2)
            image = cv2.resize(image,(48,48))
            img = extract(image)
            pred = model.predict(img)
            pred_label = labels[pred.argmax()]
            cv2.putText(im, '%s' % pred_label, (p-10, q-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("ED",im)
    except cv2.error:
        pass