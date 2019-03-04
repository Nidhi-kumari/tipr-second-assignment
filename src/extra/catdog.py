import numpy as np
import os
import cv2
import random
import pickle
import mlpfordataset as mymlp
import scipy.ndimage

import matplotlib.pyplot as mpl
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

a = input("Enter 1 for cat-dog, 2 for mnist")
if (int(a) == 1):
    #datadir = "/storage2/home2/e1-313-15521/tipr-second-assignment/data/Cat-Dog/"
    datadir = '../data/Cat-Dog'
    categs = ["cat", "dog"]
    categslabel = ["0","1"]
else:
    #datadir = "/storage2/home2/e1-313-15521/tipr-second-assignment/data/MNIST/"
    datadir = '../data/MNIST'
    categs = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    categslabel = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

X = []
label = []
for categ in categs:
    i = 0
    path = os.path.join(datadir,categ)
    label.append(int(categslabel[i]))
    i += 1
    Xlabel = []
    #print("hello 34")
    for img in os.listdir(path):
        #imgarray =  cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #colour is not a differentiating factor
        imgarray = scipy.ndimage.imread(os.path.join(path,img))
        
        imglinear = []
        for i in range(len(imgarray)):
            for j in range(len(imgarray[0])):
                imglinear.append(imgarray[i][j])
        Xlabel.append(imglinear)
    X.append(Xlabel)
print("hello")	
labeltrain = []
trainingSet = []
labeltest = []
testSet = []
for labelclass in range(len(X)):
    for trainclass in range(len(X[int(labelclass)])):
        if random.random() < 0.7:
            trainingSet.append(X[labelclass][trainclass])
            labeltrain.append(labelclass)
        else:
            testSet.append(X[labelclass][trainclass])
            labeltest.append(labelclass)
newtrain = []
newlabeltrain = []
u = []
for i in range(len(trainingSet)):
    for l1 in range(1000): #make this huge number in actual dataset
        x1 = random.randint(0, len(trainingSet) - 1)
        if (x1 in u):
            continue
        else:
            u.append(x1)
            newtrain.append(trainingSet[x1])
            newlabeltrain.append(labeltrain[x1])
X2 = np.array(newtrain)/255
y11 = np.zeros(([len(newtrain),1]))
for i in range(len(newtrain)):
    y11[i][0] = newlabeltrain[i]
yl = categslabel
X2test = np.array(testSet)/255
y11test = labeltest
'''
y1t = np.zeros(([len(testSet),1]))
for i in range(len(testSet)):
    y1t[i][0] = labeltest[i]
'''

'''
X2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y11 = [[0], [1], [1], [1]]
yl = ["0","1"]
X2test = X2
y11test = y11
'''

numlayer1 = input("Enter number of hidden layers")
numneuron1 = list(input("Enter number of neurons in each hidden layer as list")[1:-1].split(" "))
numlayer = int(numlayer1) + 2
print(numlayer)
numneuron = []
numneuron.append(int(len(X2[0]))) #[0]
for i in range(len(numneuron1)):
    numneuron.append(int(numneuron1[i]))
numneuron.append(int(len(yl)))
layeractivfunc1 = list(input("Enter activation function in each hidden layer")[1:-1].split(" "))

layeractivfunc = []
layeractivfunc.append("buffer")

for i in range(len(layeractivfunc1)):
    layeractivfunc.append(layeractivfunc1[i])
layeractivfunc.append("softmax")
print(layeractivfunc)


mymlp.mlpmain(numlayer, numneuron, layeractivfunc, X2, y11, X2test, y11test)
