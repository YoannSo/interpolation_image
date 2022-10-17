# Ne pas mettre ce code sur un espace public.

import os
import sys
import glob
import numpy as np
import tensorflow as tf
from datetime import datetime
from data_manager import DataManager
from tqdm import tqdm

from absl import app
from absl import flags

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE

from model import buildModel
from random import randint

# Voir
# https://github.com/alexbooth/DAE-Tensorflow-2.0

flags.DEFINE_integer("epochs", 10, "number of epochs")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_string("logdir", "./tmp/log", "log file directory")
flags.DEFINE_boolean("keep_training", False, "continue training same weights")
FLAGS = flags.FLAGS

model_path = None

def train(model):
    dm = DataManager()
    
    n_epochs = FLAGS.epochs

    loss = None
    for epoch in range(n_epochs):
        print('Epoch', epoch, '/', 'n_epochs')
        #data, labels = dm.get_batch(FLAGS.batch_size, for_training=True)
        data, labels = dm.get_batch(FLAGS.batch_size, 0)
        """ Pour toutes les images, pour tout les pixels, ont creer un patch et on les envoies au model """
        patch_liste = []
        width = len(data[0][0])
        height = len(data[0][0][0])
        res=[]
        # Pour les images du batch
        for k in range(FLAGS.batch_size):            
            for r in range(10):
                x = randint(0,width-1)
                y = randint(0,height-1)
                patch1=[]
                patch2=[]
                res.append(labels[k][x][y])
                tmp=0
                #on charge les patchs 
                for i in range (x-39,x+40):
                    patch1.append([])
                    patch2.append([])
                    #patch_res.append([])
                    for j in range(y-39,y+40):
                        if(i<0 or j<0 or i>=width or j>=height):
                            patch1[tmp].append([0,0,0])
                            patch2[tmp].append([0,0,0])
                        else:
                            patch1[tmp].append(data[k][0][i][j])
                            patch2[tmp].append(data[k][1][i][j])
                    #patch_res[tmp].append(labels[k][x][y])
                    tmp+=1
                temp=[]
                temp.append(patch1)
                temp.append(patch2)
                
                patch= np.array(temp).reshape(79,79,6)
                patch_liste.append(patch)
        
        patch_liste = np.array(patch_liste)
        res = np.array(res)
        print(patch_liste.shape)
        print(res.shape)

        loss = model.train_on_batch(patch_liste, res)
        print("Epoch {} - loss: {}".format(epoch, loss))
        model.save(model_path)
    print("Finished training.")

def load_model():
    """Set up and return the model."""
    model = buildModel()
    optimizer = Adam(FLAGS.learning_rate)
    loss = MSE
    metrics=['accuracy']
    # load most recent weights if model_path exists 
    if os.path.isfile(model_path):
        print("Loading model from", model_path)
        model.load_weights(model_path)

    model.compile(optimizer, loss,metrics)
    model.summary()
    return model

def setup_paths():
    """Create log and trained_model dirs. """
    global model_path, summary_path
    os.makedirs(FLAGS.logdir, exist_ok=True)
    os.makedirs("./trained_model", exist_ok=True)
    timestamp = 'timestamp' # str(datetime.now())

    if FLAGS.keep_training and os.listdir(FLAGS.logdir):
        files = filter(os.path.isdir, glob.glob(FLAGS.logdir + "/*"))
        files = sorted(files, key=lambda x: os.path.getmtime(x))
        timestamp = os.path.basename(os.path.normpath(list(reversed(files))[0]))

    model_path = os.path.join("./trained_model/DAE-model-" + timestamp + ".h5")
    summary_path = os.path.join(FLAGS.logdir, timestamp)

def main(argv):
    setup_paths()
    model = load_model()
    train(model)

if __name__ == '__main__':
    app.run(main)
