from uuid import uuid4

from flask import Flask, request, render_template, send_from_directory,redirect

from werkzeug import SharedDataMiddleware

# requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template, request
# scientific computing library for saving, reading, and resizing images
from scipy.misc import imread, imresize
# for matrix math
import numpy as np
# for regular expressions, saves time dealing with string data
import re
# system level operations (like loading files)
import sys
# for reading operating system data
import os
# tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from model.load import *
# initalize our flask app

#UPLOAD_FOLDER = 'upload'
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# global vars for easy reusability
global model, graph

# initialize these variables
model, graph = init()


import sys,os,fnmatch
import base64

# import the necessary packages for image classification
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

import inspect
#import tkinter as tk
from tkinter import filedialog

from os.path import dirname, abspath
from filedialog import *


#from bs4 import BeautifulSoup
#from requests_html import HTMLSession


#from selenium import webdriver
#from selenium.webdriver.common.by import By

import pathlib

from pathlib import Path




app = Flask(__name__)

#from api import controller

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/mnist/')
def mnist():
    return render_template('mnist.html')

@app.route('/image_recognition/')
def image_recognition():
    return render_template('image_recognition.html')

@app.route('/workinprogress1/')
def workinprogress1():
    return render_template('workinprogress1.html')

@app.route('/workinprogress2/')
def workinprogress2():
    return render_template('workinprogress2.html')

@app.route('/workinprogress3/')
def workinprogress3():
    return render_template('workinprogress3.html')

# decoding an image from base64 into raw representation
def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))
        
        


@app.route('/predict_digit/', methods=['GET', 'POST'])
def predict_digit():
    # whenever the predict method is called, we're going
    # to input the user drawn character as an image into the model
    # perform inference, and return the classification
    # get the raw data format of the image
    imgData = request.get_data()
    # encode it into a suitable format
    convertImage(imgData)
    # read the image into memory
    x = imread('output.png', mode='L')
    # make it the right size
    x = imresize(x, (28, 28))
    # imsave('final_image.jpg', x)
    # convert to a 4D tensor to feed into our model
    x = x.reshape(1, 1, 28, 28)
    # in our computation graph
    with graph.as_default():
        # perform the prediction
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        # convert the response to a string
        response = np.argmax(out, axis=1)
        return str(response[0])

APP_ROOT = os.path.dirname(os.path.abspath(__file__))



def browse_file():
    home = str(Path.home())
    #original_path = os.getcwd()
    second_path = os.path.join(home,'Image')
    #filenames = os.listdir('Image')
    filename = os.listdir(second_path)
    file_path = os.path.join(second_path,filename[0])
    #abspath = os.path.abspath(path)
    return file_path








@app.route('/predict_image/', methods=['POST','GET']) # Add the correct route
def predict_image():

    file_name = browse_file()

    image = image_utils.load_img(file_name, target_size=(224, 224)) #Enter correct file path for the uploaded image
    image = image_utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    model = VGG16(weights="imagenet")
    preds = model.predict(image)
    P = decode_predictions(preds)
    # loop over the predictions and display the rank-5 predictions +
    # probabilities to our terminal
    for (i, (imagenetID, label, prob)) in enumerate(P[0]):
        print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
        orig = cv2.imread(file_name)
        (imagenetID, label, prob) = P[0][0]
        cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        #plt.imshow(orig)
        #plt.show()
        #cv2.imshow("Classification", orig)
        #cv2.waitKey(0)
    return label



@app.route("/upload", methods=['POST', 'GET'])
def upload():
    target = os.path.join(APP_ROOT, 'imagesPrediction')
    #target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
    
    return render_template("complete_display_image.html", image_name=filename)




@app.route('/upload/<filename>')
def send_image(filename):

    return send_from_directory("imagesPrediction", filename)



app.add_url_rule('/upload/<filename>', 'send_image',build_only=True)




if __name__ == "__main__":

    app.run(host='0.0.0.0')


