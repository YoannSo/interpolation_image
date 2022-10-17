# Ne pas mettre ce code sur un espace public.

import numpy as np
import tensorflow as tf
import cv2
import os
from numpy import float32
class DataManager:
     # Ici, vous pouvez lire et pré-analyser la video utilisee pour le training
        # Cette video doit etre unique.
        # Notamment, un apprentissage judicieux (evoque en session) vous permettra potentiellement de mieux
        # rentabiliser l'apprentissage.
        
        # Si ce drapeau est defini, votre code ne devra prendre en compte que les 10 premieres images de la video
        # pour l'apprentissage. Cette fonctionnalite est obligatoire (voir sujet du projet).
        
    def __init__(self):
        self.X = []
        self.training_set_size = None
        self.load_data() 
    
    def load_data(self):
        dataSet=[]
        test=[]
        #path = './data/img_align_celeba/img_align_celeba/' 
        path = './data/' 
        listVideo = os.listdir(path)
        
        for videoName in listVideo:
            video=cv2.VideoCapture(path+videoName)
            while(video.isOpened()):
                ret,frame=video.read()
                if(ret==False):
                    break;
                frame = cv2.resize(frame, dsize=(640, 360),interpolation= cv2.INTER_CUBIC)
                dataSet.append(frame)
        self.X = np.array(dataSet)
        self.X = self.X.astype(np.uint)        
                
       

      # Ici vous devez retourner deux valeurs, a savoir, un ensemble de donnees ou chaque donnee individuelle est:
        # - un image I1 et une image I2 (donnees d'entree)
        # - un image I^ attendue (donnee de sortie)
        # Comme evoque dans la remarque dans la fonction precedente, les donnees a renvoyer sont
        # idealement judicieusement choisies, meme si ce n'est pas strictement obligatoire.
        # La qualite de vos resultats en dependra, ainsi que la quantite de temps allouee pour l'apprentissage
        # (qui pourra vite devenir prohibitive)
        
        # Cette fonction doit (c'est obligatoire pour votre projet) retourner
        # - dans le premier membre du tuple, un batch de donnees d'entree
        # - dans le second membre, un batch des labels correspondants
        
    # batchX shape = ( nbElem, ImgAvant/Apres , height,widht,RGB) 
    def get_batch(self, batch_size, index):
        batchX=[]
        batchY=[]
        if(index==0):
            index=1
        if(index+batch_size==len(self.X)-1):
            index-=1
        for i in range(index,index+batch_size):
            batchX.append([self.X[i-1],self.X[i+1]]) 
            batchY.append(self.X[i])
        return np.array(batchX),np.array(batchY)
    
    def get_patch(self,img1, img2,y,x):
        patch_liste = []
        width = 640
        height = 360
        # Pour les images du batch
        patch1=[]
        patch2=[]
        tmp=0
        #on charge les patchs 
        for j in range(y-39,y+40):
            patch1.append([])
            patch2.append([])
            for i in range (x-39,x+40):
                if(i<0 or j<0 or i>=width or j>=height):
                    patch1[tmp].append([0,0,0])
                    patch2[tmp].append([0,0,0])
                else:
                    patch1[tmp].append(img1[j][i])
                    patch2[tmp].append(img2[j][i])
            tmp+=1
        temp=[]
        temp.append(patch1)
        temp.append(patch2)
                
        patch= np.array(temp).reshape(79,79,6)
        return patch
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        