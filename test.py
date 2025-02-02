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

cap = cv2.VideoCapture(0)
labels = {0:'angry', 1:'disgust', 2:'fear', 3: 'happy', 4 : 'neutral', 5: 'sad', 6: 'surprise'}


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame,1.4,5)
    try:
        for p,q,r,s in faces:
            image = gray[q:q+s, p:p+s]
            cv2.rectangle(frame, (p,q), (p+r, q+s), (0, 255, 0), 2)
            image = cv2.resize(image,(48,48))
            img = extract(image)
            pred = model.predict(img)
            pred_label = labels[pred.argmax()]
            cv2.putText(frame, '%s' % pred_label, (p-10, q-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("ED",frame)

    except:
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
