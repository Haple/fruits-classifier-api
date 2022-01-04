from flask import Flask, request, jsonify
from fastai.vision.all import *
from PIL import Image
from flask_cors import CORS,cross_origin
import os
app = Flask(__name__)
CORS(app, support_credentials=True)

path = Path()
learn = load_learner(path/'my-first-fruits-classifier.pkl')
classes = learn.dls.vocab

@app.route('/predict',methods=['POST'])
def predict():
    img_file = request.files['image']
    img_decoded = tensor(Image.open(img_file))
    prediction = learn.predict(img_decoded)
    probs_list = prediction[2].numpy()
    return jsonify({
        'category': classes[prediction[1].item()],
        'probs': {c: round(float(probs_list[i]), 5) for (i, c) in enumerate(classes)}
    })

if __name__ == "__main__":
    print(("Loading model..."))
    load_model()
    print(("Model loaded! Starting the app..."))
    app.run(debug=True)
