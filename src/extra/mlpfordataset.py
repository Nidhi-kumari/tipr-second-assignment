import numpy as np
import matplotlib.pyplot as mpl
import math
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def mlpmain(numlayer, numneuron, layeractivfunc, X2, y11, X2test, y11test):

    class layer():
        def __init__(self, numneuron, layeractivfunc):
            self.numneuron = numneuron
            self.vj = np.zeros([numneuron, 1])
            self.layerout = np.zeros([numneuron, 1])
            self.layeractivfunc = layeractivfunc #check where to include this
            self.localgrad = np.zeros([numneuron, 1])

    class mynn():
        def __init__(self, numlayer, numneuron):
            self.numlayer = numlayer
            self.numneuron = numneuron
            self.mynnlayers = []
            self.eta = 0.1

        def initialization(self):
            for ln in range(self.numlayer):
                eachlayer = layer(numneuron=numneuron[ln],layeractivfunc=layeractivfunc[ln])
                if (ln != self.numlayer -1):
                    sigma = (self.numneuron[ln + 1])**(-0.5)
                    eachlayer.weighttrans = np.random.normal(0, sigma, size=(self.numneuron[ln + 1], self.numneuron[ln]))
                    eachlayer.bias = 0.25 * np.ones((self.numneuron[ln + 1], 1))
                self.mynnlayers.append(eachlayer)

        def forwardpropagation(self, inputfeaturedata):
            self.mynnlayers[0].layerout = inputfeaturedata[np.newaxis]
            for ln in range(self.numlayer - 1):
                if (ln == self.numlayer - 2 ):
                    lnplus = ln + 1
                    b1 = (self.mynnlayers[ln].layerout)
                    a1 = np.add(np.matmul(self.mynnlayers[ln].weighttrans,b1.T),self.mynnlayers[ln].bias)
                    self.mynnlayers[int(lnplus)].vj = np.add(np.matmul(self.mynnlayers[ln].weighttrans,b1.T),self.mynnlayers[ln].bias).T
                    self.mynnlayers[int(lnplus)].layerout = self.activfunc(a1.T, 'softmax') #'softmax'
                else:
                    lnplus = ln + 1
                    b = (self.mynnlayers[ln].layerout) #[np.newaxis]
                    a = (np.add(np.matmul(self.mynnlayers[ln].weighttrans,b.T),self.mynnlayers[ln].bias))
                    self.mynnlayers[int(lnplus)].vj = (np.add(np.matmul(self.mynnlayers[ln].weighttrans,b.T),self.mynnlayers[ln].bias)).T
                    self.mynnlayers[int(lnplus)].layerout = self.activfunc(a.T , self.mynnlayers[lnplus].layeractivfunc)


        def backpropagation(self, yhat1, correctcount, totalcount, corriniter, totiniter, tocalf1):
            yhat = np.zeros([numneuron[self.numlayer - 1],1])
            yhat[yhat1][0] = 1
            error =  yhat - self.mynnlayers[self.numlayer - 1].layerout.T
            deriv = self.activfuncderiv(self.mynnlayers[self.numlayer - 1].vj.T, self.mynnlayers[self.numlayer - 1].layeractivfunc)
            self.mynnlayers[self.numlayer - 1].localgrad = np.multiply(error, deriv)
            for ln in range(self.numlayer - 2, 0 , -1):
                lnplus = ln +1
                yout = self.mynnlayers[ln].layerout.T
                yout1 = self.mynnlayers[ln].vj.T
                derivln = self.activfuncderiv(yout1, self.mynnlayers[ln].layeractivfunc)
                a = np.matmul(self.mynnlayers[lnplus].localgrad.T, self.mynnlayers[ln].weighttrans)
                self.mynnlayers[ln].localgrad = np.multiply(a.T, derivln)
            for ln in range(self.numlayer -1):
                lnplus = ln + 1
                deltaw = self.eta * np.matmul(self.mynnlayers[lnplus].localgrad, self.mynnlayers[ln].layerout)
                self.mynnlayers[ln].weighttrans += deltaw
                deltab = self.eta * self.mynnlayers[lnplus].localgrad
                self.mynnlayers[ln].bias += deltab
            totalcount+=1
            totiniter +=1
            tocalf1.append(np.argmax(self.mynnlayers[self.numlayer - 1].layerout))
            if(np.argmax(self.mynnlayers[self.numlayer - 1].layerout) == yhat1):
                correctcount += 1
                corriniter +=1
            return correctcount, totalcount, corriniter, totiniter, tocalf1

        def activfunc(self, indata, layeractivfunc):
            if (layeractivfunc == "sigmoid"):
                return 1/(1 + np.exp(-indata))
            elif (layeractivfunc == "tanh"):
                return (np.exp(indata) - np.exp(-indata))/(np.exp(indata) + np.exp(-indata))
            elif (layeractivfunc == "swish"):
                return 1 / (1 + np.exp(-np.multiply(10,indata)))
            elif (layeractivfunc == "relu"):
                return 1 / (1 + np.exp(-np.multiply(10000,indata)))
            elif (layeractivfunc == "buffer"):
                return indata
            elif (layeractivfunc == "softmax"):
                return np.exp(indata) / np.sum(np.exp(indata))
            else:
                pass

        def activfuncderiv(self, indata, layeractivfunc):
            if (layeractivfunc == "sigmoid"):
                y = (1 / (1 + np.exp(-indata)))
                return y * (1-y)
            elif (layeractivfunc == "tanh"):
                y = (np.exp(indata) - np.exp(-indata))/(np.exp(indata) + np.exp(-indata))
                return 1- np.power(y,2)
            elif (layeractivfunc == "swish"):
                y = 1 / (1 + np.exp(-np.multiply(10,indata)))
                return 10 * y * (1 - y)
            elif (layeractivfunc == "relu"):
                y = 1 / (1 + np.exp(-np.multiply(10000, indata)))
                return 10000 * y * (1 - y)
            elif (layeractivfunc == "softmax"):
                y = np.exp(indata)/np.sum(np.exp(indata))
                return y * (1-y)
            else:
                pass

        def forwardpropagationtest(self, inputfeaturedata):
            self.mynnlayers[0].layerout = inputfeaturedata[np.newaxis]
            for ln in range(self.numlayer - 1):
                if (ln == self.numlayer - 2):
                    lnplus = ln + 1
                    b1 = (self.mynnlayers[ln].layerout)
                    a1 = np.add(np.matmul(self.mynnlayers[ln].weighttrans, b1.T), self.mynnlayers[ln].bias)
                    self.mynnlayers[int(lnplus)].layerout = self.activfunc(a1.T, 'softmax')
                else:
                    lnplus = ln + 1
                    b = (self.mynnlayers[ln].layerout)
                    a = (np.add(np.matmul(self.mynnlayers[ln].weighttrans, b.T), self.mynnlayers[ln].bias))
                    self.mynnlayers[int(lnplus)].layerout = self.activfunc(a.T, self.mynnlayers[lnplus].layeractivfunc)
            return self.mynnlayers[self.numlayer - 1].layerout


    myneuralnetwork = mynn(numlayer, numneuron)
    myneuralnetwork.initialization()
    correctcount = 0
    totalcount = 0
    correcttest = 0
    totaltest = 0
    for epoch in range(100):
        X1 = np.zeros([len(X2), len(X2[0])])
        y = []
        u = []
        for ln in range(len(X1)):
            for l1 in range(1000):
                i = random.randint(0, len(X1) - 1)
                if (i in u):
                    continue
                else:
                    u.append(i)
                    X1[ln] = X2[i]
                    y.append(y11[i])
                    break
        ylabel = []
        tocalf1 = []
        for num in range(len(y)): #changed from X1 to y
            ylabel.append(y[num][0])
            myneuralnetwork.forwardpropagation(X1[num])
            corriniter = 0
            totiniter = 0
            correctcount, totalcount, corriniter, totiniter, tocalf1  = myneuralnetwork.backpropagation(int(y[num][0]), correctcount, totalcount, corriniter, totiniter, tocalf1)
        print('Classification Report: ', classification_report(ylabel, tocalf1))
        acc = corriniter/totiniter
    print("train accuracy= ", correctcount/totalcount)
    print("Testing")
    for i in range(len(X2test)):
        a = myneuralnetwork.forwardpropagationtest(X2test[i])
        totaltest += 1
        if (y11test[i] == np.argmax(a)):
            correcttest +=1
    print("test accuracy is :", correcttest/totaltest)

