from flask import Flask, render_template, Response
import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier


app = Flask(__name__)
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20

def gen_frames():
    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)
        prediction_text = ''

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((300, 300, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            aspectRatio = h / w
            if aspectRatio > 1:
                k = 300 / h
                wCalc = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCalc, 300))
                wGap = math.ceil((300 - wCalc) / 2)
                imgWhite[:, wGap:wGap + wCalc] = imgResize
            else:
                k = 300 / w
                hCalc = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (300, hCalc))
                hGap = math.ceil((300 - hCalc) / 2)
                imgWhite[hGap:hGap + hCalc, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite)
            prediction_text = str(prediction)

        cv2.putText(img, prediction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
