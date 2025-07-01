import base64
from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import math
import os
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# import os to set env variable
import os

app = Flask(__name__)
cap = None
streaming = False

detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def generate_frames(frameData):
    if not frameData:
        return {"success": False, "message": "No frame data provided."}

    if ',' in frameData:
        image_data = frameData.split(',')[1]
    else:
        image_data = frameData

    try:
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return {"success": False, "message": "Error decoding image."}

    if img is None:
        # Return a blank frame or handle error
        img = np.ones((300, 300, 3), np.uint8) * 255

    hands, img = detector.findHands(img)
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
    return buffer.tobytes()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/imp')
def imp():
    return render_template('imp.html')

@app.route('/video', methods=['POST'])
def video():
    # global streaming
    # streaming = True
    frameData = request.json.get('image')
    if not frameData:
        return {"error": "No frame data provided"}, 400
    
    img_bytes = generate_frames(frameData)
    return Response(img_bytes, mimetype='image/jpeg')

@app.route('/shutdown')
def shutdown():
    global streaming
    streaming = False
    return "Stream stopped"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)

