from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model, model_from_json
import joblib

IMAGE_FOLDER = os.path.join("static", "images")

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = IMAGE_FOLDER

def init():
    global model_bell, model_lstm, graph
    model_bell = joblib.load('../model/ovr_logistic.joblib')
    model_lstm = load_model('../model/lstm_sentiment_analysis.h5')
    model_json = model_lstm.to_json()
    with open("../model/lstm_sentiment_analysis.json", "w") as json_file:
        json_file.write(model_json)
        json_file.close()
    # model_lstm.save_weights("color_tensorflow_real_mode_2_imgs.h5")
    graph = tf.compat.v1.get_default_graph()

@app.route("/", methods=["GET", "POST"])
def home():
    return "<h1>hello WeBell-App users<h1>"

@app.route("/health")
def health():
    return "<h3>Yes, I am Healthy and ALIVE! Thanks for checking!   --   My name is WeBell App btw! :) </h3>"

@app.route("/predict")
def predict_home():
    return f"<h1>You can ask me sentiments for your texts by simply going to \"/predict_tools\" url OR by typing your text in front of the url \"/predict/your text\" </h1>"

@app.route("/predict/<text>", methods=["GET", "POST"])
def predict(text):
    if request.method=='POST':
        text = request.form['text']
        sentiment = ''
        max_review_length = 500
        word_to_id = imdb.get_word_index()
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        text = text.lower().replace("<br />", " ")
        text=re.sub(strip_special_chars, "", text.lower())
    result = model_bell.predict([text])
    return f"<h1>you wanted to know the sentiments for \"{text}\"    -->    Based on \"Bell Provided Model\", this is your result = {result}</h1>"

@app.route("/predict_tools", methods=["GET", "POST"])
def predict_tools():
    text = ""
    sentiment = ""
    probability = ""
    img_filename = ""

    if request.method=='POST':
        text = request.form['text']
        max_review_length = 500
        word_to_id = imdb.get_word_index()
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        text = text.lower().replace("<br />", " ")
        text=re.sub(strip_special_chars, "", text.lower())

        words = text.split() #split string into a list
        x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=20000) else 0 for word in words]]
        x_test = sequence.pad_sequences(x_test, maxlen=500) # Should be same which you used for training data
        vector = np.array([x_test.flatten()])
        with graph.as_default():
            json_file = open('../model/lstm_sentiment_analysis.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("../model/lstm_sentiment_analysis.h5")
            # loaded_model.compile()
            # model_lstm.call = tf.function(model_lstm.call)
            probability = loaded_model.predict(array([vector][0]))[0][0]
            class1 = loaded_model.predict_classes(array([vector][0]))[0][0]
        if class1 == 0:
            sentiment = 'Negative'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sad.png')
        else:
            sentiment = 'Positive'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'happy.png')
    return render_template('home.html', text=text, sentiment=sentiment, probability=probability, image=img_filename)

if __name__ == "__main__":
    init()
    app.run(debug=True)