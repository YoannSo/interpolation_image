# Ne pas mettre ce code sur un espace public.

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D,Conv3D, Conv2DTranspose, Dense, Flatten, Reshape, BatchNormalization
from warnings import filters

def getKernel(inputs):
    
    #conv + BN
    X_pred=(Conv2D(kernel_size=(7,7), filters=32, padding="valid", activation="relu")(inputs))
    X_pred=BatchNormalization()(X_pred)
    #down_conv
    X_pred=Conv2D(kernel_size=(2,2), filters=32, strides=(2,2), padding="valid", activation="relu")(X_pred)
    
    #conv + BN
    X_pred=Conv2D(kernel_size=(5,5), filters=64, padding="valid", activation="relu")(X_pred)
    X_pred=BatchNormalization()(X_pred)
    #down
    X_pred=Conv2D(kernel_size=(2,2), filters=64, strides=(2,2), padding="valid", activation="relu")(X_pred)
    
    #conv + BN
    X_pred=Conv2D(kernel_size=(5,5), filters=128, padding="valid", activation="relu")(X_pred)
    X_pred=BatchNormalization()(X_pred)
    #down
    X_pred=Conv2D(kernel_size=(2,2), filters=128, strides=(2,2), padding="valid", activation="relu")(X_pred)
    
    #conv + BN
    X_pred=Conv2D(kernel_size=(3,3), filters=128,padding="valid", activation="relu")(X_pred)
    X_pred=BatchNormalization()(X_pred)
    
    #conv
    X_pred=Conv2D(kernel_size=(4,4), filters=2048, strides=(1,1), padding="valid", activation="relu")(X_pred)
    
    X_pred=Conv2D(kernel_size=(1,1), filters=3362, strides=(1,1), padding="valid", activation="softmax")(X_pred)
    
    X_pred = Reshape((41,82,1))(X_pred)
    return X_pred
    
    
def buildPixel(kernel,inputs):
    image_tensors=tf.split(inputs[:,19:60,19:60,:],num_or_size_splits=2,axis=-1)
    kernel_tensors=tf.split(kernel,num_or_size_splits=2,axis=-2)
    img1=image_tensors[0]
    img2=image_tensors[1]
    output1=img1*kernel_tensors[0]
    output2=img2*kernel_tensors[1]
    pixel1=tf.math.reduce_sum(output1,axis=[1,2],keepdims=True)
    pixel2=tf.math.reduce_sum(output2,axis=[1,2],keepdims=True)
    
    pixel=tf.squeeze(pixel1,axis=[1,2])+tf.squeeze(pixel2,axis=[1,2])
    return pixel

def buildModel():
    # Le reseau ici est a creer de toutes pieces en suivant les instructions dans le papier.
    # Rappelez-vous que le plus important consiste a definir les entrees et les sorties.
    # Les details du reseau (convolutions, activations...) peuvent etre ajustes plus tard si necessaire.
    X = tf.keras.Input(shape=(79,79,6))
    X_pred = getKernel(X)
    res = buildPixel(X_pred,X)
    return tf.keras.Model(inputs=X, outputs=res, name="model")

