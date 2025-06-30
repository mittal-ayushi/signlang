from flask import Flask, render_template, Response
import cv2
import numpy as np
import math
import os
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

app = Flask(__name__)
cap = None
streaming = False

detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def generate_frames():
    global cap, streaming
    cap = cv2.VideoCapture(0)

    while streaming:
        success, img = cap.read()
        if not success:
            break

        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((300, 300, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            if imgCrop.size == 0:
                continue

            aspectRatio = h / w
            if aspectRatio > 1:
                k = 300 / h
                wCalc = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCalc, 300))
                wGap = math.ceil((300 - wCalc) / 2)
                imgWhite[:, wGap:wCalc + wGap] = imgResize
            else:
                k = 300 / w
                hCalc = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (300, hCalc))
                hGap = math.ceil((300 - hCalc) / 2)
                imgWhite[hGap:hCalc + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite)
            cv2.putText(img, labels[index], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    if cap is not None:
        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/imp')
def imp():
    return render_template('imp.html')

@app.route('/video')
def video():
    global streaming
    streaming = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown')
def shutdown():
    global streaming
    streaming = False
    return "Stream stopped"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)


