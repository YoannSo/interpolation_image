# Ne pas mettre ce code sur un espace public.

import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from data_manager import DataManager
from sklearn.feature_extraction.image import extract_patches_2d
import matplotlib.pyplot as plt 

from model import buildModel

from absl import app
from absl import flags
from pip._vendor.requests.api import patch
from tensorflow.python.eager.function import np_arrays

import math

flags.DEFINE_string("model", "./trained_model/DAE-model-timestamp.h5", "Path to a trained model (.h5 file)")
FLAGS = flags.FLAGS

def predict(model):
    dm = DataManager()
    #video  = cv2.VideoCapture('input_video.mp4')
    video  = cv2.VideoCapture('data/car001.mp4')

    # Recupere le nombre d'images par seconde et la dimension    
    fps = video.get(cv2.CAP_PROP_FPS)
    width = video .get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video .get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width),int(height))
    # Nous allons generer une video avec 2 fois plus d'images par seconde
    # CV_FOURCC_PROMPT peut etre utilise pour demander a choisir le format
    # de sortie parmi une liste. On choisira un format de sortie *sans perte* (lossless, uncompressed)
    # fourcc = CV_FOURCC_PROMPT
    # fourcc = cv2.VideoWriter_fourcc(*'HFYU') # Huffman Lossless Codec
    #fourcc = cv2.VideoWriter_fourcc(*'XVID') # ou *MJPG
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #out = cv2.VideoWriter(['output_video.avi', fourcc, fps*2, (width, height),True])    
    #vidwrite = cv2.VideoWriter(['testvideo', fourcc, 50, (640,480)])
    writer = cv2.VideoWriter("data/testvideo.mp4",fourcc, fps*2.0,size) 
    
    
    # On lit la video d'entree tant qu'il y a des images a lire
    hasImages = True
    while hasImages:
        hasImages, img1 = video.read()
        if hasImages == True:
            # Redimensionner a la resolution voulue
            img1= cv2.resize(img1, (640, 360))
            #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            hasImages, img2 = video.read()
            if hasImages == True:
                # Redimensionner a la resolution voulue
                img2= cv2.resize(img2, (640, 360))
                #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                
                # A ce stade, nous avons deux images, et nous souhaitons generer (predire) une image interpolee I^
                
                imgHat = np.zeros_like(img1) # I^
                print(img1.shape)
                sizey = img1.shape[0] #360
                sizex = img1.shape[1] #640
                
                # On traite tous les pixels
                patchs=[]
                for y in range(sizey):
                    print("y")
                    print(y)
                    for x in range(sizex):
                        """ Methode d'origine mais malheuresement trop lourd en temps de calcul"""
                        # patchs.append(dm.get_patch(img1,img2,y,x))
                        """ Methode revisite """
                        
                        
                       
                patchs= np.array(patchs)
                print(patchs.shape)
                """
                imgHat =model.predict(patchs)
                writer.write(img1)
                writer.write(imgHat)  
                """
        hasImages = False   
    writer.release()
    video.release()
    
def predictImage(model,img1,img2):
    frame1 = cv2.imread(img1)
    frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB)
    save1 = frame1
    
    frame2 = cv2.imread(img2)
    frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    save2 = frame2
    
    myprediction= predict(model,frame1,frame2,frame1.shape[0],frame1.shape[1])
    
def predict(model,frame1,frame2,height,width):
    Frame1=tf.pad(frame1,[[39,39],[39,39],[0,0]],"CONSTANT")
    Frame2=tf.pad(frame2,[[39,39],[39,39],[0,0]],"CONSTANT")
    
    Frame3 = np.concatenate((Frame1,Frame2),axis=-1)
    
    prediction = np.empty((height,width,3),dtype="uint8")
    w=math.ceil(width/20)
    j=0
    while j<width-w:
        prediction[:,j:w+j,:]=np.reshape(model.predict(extract_patches_2d(Frame3[:,j:j+w-1+79,:],(79,79)),batch_size=128,use_multiprocessing=True),(height,w,3))
        j=j+w
    if j!=width:
        w=width-j
        prediction[:,j:w+j,:]=np.reshape(model.predict(extract_patches_2d(Frame3[:,j:j+w-1+79,:],(79,79)),batch_size=128,use_multiprocessing=True),(height,w,3))
    return prediction        

def load_model():
    """Set up and return the model."""
    model = buildModel()
    model_path = os.path.abspath(FLAGS.model)
    if os.path.isfile(model_path):
        print("Loading model from", model_path)
        model.load_weights(model_path)
    return model
       
def main(argv):
    if FLAGS.model == None:
        print("Please specify a path to a model with the --model flag")
        sys.exit()
    model = load_model()
    video  = cv2.VideoCapture('data/car001.mp4')
    hasImages, img1 = video.read()
    img1= cv2.resize(img1, (640, 360))
    hasImages, img2 = video.read()
    img2= cv2.resize(img2, (640, 360))
    #predict(model)
    res=predict(model, img1, img2, 360, 640)
    plt.imsave("frameMid.jpg",res)

if __name__ == '__main__':
    app.run(main)
    
                
