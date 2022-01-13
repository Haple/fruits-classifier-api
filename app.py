from flask import Flask, request, jsonify, render_template
from fastai.vision.all import *
from PIL import Image
from flask_cors import CORS,cross_origin
import os
app = Flask(__name__)
CORS(app, support_credentials=True)

learn = load_learner(Path()/'my-first-fruits-classifier.pkl')
classes = learn.dls.vocab

def predict(img_file):
    img_decoded = tensor(Image.open(img_file))
    prediction = learn.predict(img_decoded)
    probs_list = prediction[2].numpy()
    return {
        'category': classes[prediction[1].item()],
        'probs': {c: round(float(probs_list[i]), 5) for (i, c) in enumerate(classes)}
    }

@app.route('/',methods = ['POST', 'GET'])
def predict_html():
    if request.method == 'POST':
        img_file = request.files['img']
        return render_template("index.html", result = predict(img_file))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_json():
    img_file = request.files['image']
    return jsonify(predict(img_file))

if __name__ == "__main__":
    app.run(debug=True)
