#!/usr/bin/env python
# coding: utf-8




from matplotlib import pyplot as plt
from random import randint
import os
from matplotlib.pyplot import imshow
import scipy.ndimage
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pickle
from sklearn.metrics import f1_score
from sklearn import preprocessing



def sigmoid(X):
    return 1/(1 + np.exp(-X));

def relu(X):
    return np.maximum(0,X);

def swish(x,y):
    return x*y;

def derv_sigmoid(x):
    return x*(1-x);

def derv_relu(x):
    sh = x.shape;
    m = sh[0];
    n = sh[1];
    z = np.ones((m,n));
    z = z*(x>0);
    return z;

def derv_swish(x,y):
    return (x+y*(1-x));



def forwardPropogation (X,Y,netConfig,batch_size,lr,total_epochs,act = "sigmoid"):
        
    sh = X.shape;
    m = sh[0]; #number of examples.
    n = sh[1]; #number of features.
    
    X0 = np.ones((m,1))
    X_new = np.hstack((X,X0))
    X_new = np.transpose(X_new);
    
    Y_new = np.transpose(Y);
    
    #Initialize Weights
    wt = {};
    total_layers = len(netConfig);
    wt_layers= total_layers - 1;
    
    for i in range(wt_layers):
        if(i == 0):
            wt["wt_hidden"+str(i)] = np.random.uniform(-5.0,5.0,[netConfig[i + 1], netConfig[i] +1])/n;
        else:
            wt["wt_hidden"+str(i)] = np.random.uniform(-5.0,5.0,[netConfig[i + 1], netConfig[i] +1]);
    
    params = {};
    act_layer = {};
    act_layer_bias = {};
    #act_layer_bias["hidden_output_bias0"] = X_new;
    
    A = np.vstack((X_new,Y_new));
    for epoch in range(total_epochs):
        
        Ap = A[:,np.random.randint(A.shape[1], size=batch_size)];
        X_new1 = Ap[0:n+1,:];
        Y_new1 = Ap[n+1:Ap.size,:]
        #print(X_new1);
        #print("forward propogation")
        #print(Y_new1)
        act_layer_bias["hidden_output_bias0"] = X_new1;
        
        for i in range(wt_layers):
            prev_wt = wt["wt_hidden"+str(i)];
            prev_ho =  act_layer_bias["hidden_output_bias"+str(i)];
            hidden_input = np.matmul(prev_wt,prev_ho);
            
            if(i+1 < wt_layers):
                
                if(act == "sigmoid"):
                    hidden_output1 = sigmoid(hidden_input);
                
                elif(act == "swish"):
                    hidden_output1 = sigmoid(hidden_input);
                    act_layer["sigmoid_output"+str(i+1)] = hidden_output1;
                    hidden_output1 = swish(hidden_input,hidden_output1);
                
                elif(act == "relu"):
                    act_layer["hidden_input"+str(i+1)] = hidden_input;
                    hidden_output1 = relu(hidden_input);
    
                hidden_output = np.vstack((hidden_output1,np.ones((1,batch_size)))); #p+1Xm
                act_layer_bias["hidden_output_bias" + str(i+1)] = hidden_output;
            else:
                hidden_output1 = sigmoid(hidden_input);
                #print(hidden_output1);
            act_layer["hidden_output"+str(i+1)] = hidden_output1;
            #print("hidd")
            #print(hidden_output1)
        wt = backwardPropogation(wt,netConfig,act_layer,act_layer_bias,Y_new1,(lr/batch_size),act);
    
    params["weights"] = wt;
    
    return params;
        
def backwardPropogation (wt,netConfig,act_layer,act_layer_bias,Y_new,alpha_prime,act): 
    
    Delta ={};
    total_layers = len(netConfig);
    wt_layers= total_layers - 1;
    
    
    fo = act_layer["hidden_output"+str(wt_layers)];
    #print("backward")
    #print(fo)
    delta_output = (fo - Y_new); #dXm matrix
    #delta_output = np.multiply((fo-Y_new),derv_sigmoid(fo));
    Delta["delta"+str(wt_layers)] = delta_output;
    
    for i in range(wt_layers-1,0,-1):
        delta_next = Delta["delta"+str(i+1)];
        wt_current = wt["wt_hidden"+str(i)];
        activation_current = act_layer["hidden_output"+str(i)];
            
        delta_current = np.matmul(np.transpose(wt_current),delta_next);
        delta_current = np.delete(delta_current,netConfig[i],0);
        
        if(act == "sigmoid"):
            delta_current = np.multiply(delta_current,derv_sigmoid(activation_current));
        
        elif(act == "swish"):
            sigmoid_current = act_layer["sigmoid_output"+str(i)];
            delta_current = np.multiply(delta_current,derv_swish(activation_current,sigmoid_current));
        
        elif(act == "relu"):
            activation_input = act_layer["hidden_input"+str(i)]; 
            delta_current = np.multiply(delta_current,derv_relu(activation_input));
        
        Delta["delta"+str(i)] = delta_current;
            
    for i in range (0,wt_layers):
        weight = wt["wt_hidden"+str(i)];
        delta_next = Delta["delta"+str(i+1)];
        activation_current = act_layer_bias["hidden_output_bias"+str(i)];
            
        weight = weight - (alpha_prime)*np.matmul(delta_next,np.transpose(activation_current));
        wt["wt_hidden"+str(i)] = weight;   
    
    
    return wt;




    
def trainNeuralNet_unified (X, Y, netConfig,batch_size,lr,epochs,act):
    # possible values of actFunc are 'sigmoid', 'ReLU', and 'sigmoid'
    
    params = {};
    total_layers = len(netConfig);
    
    if(netConfig[total_layers - 1] == 1):
         params = forwardPropogation(X,Y,netConfig,batch_size,lr,epochs,act);
    else:
        enc = preprocessing.OneHotEncoder()
        enc.fit(Y)  
        target = (enc.transform(Y).toarray());
        #print(target.shape);
        params = forwardPropogation(X,target,netConfig,batch_size,lr,epochs,act)
    '''   
    f = open("CatDogWeights", "wb")
    pickle.dump(params,f)
    f.close()
    '''
    
    
    
    return params;

    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def predictNeuralNet_unified (X_test,netConfig,params,act = "sigmoid"):
    
    wt = params["weights"];    
    
    sh = X_test.shape;
    m = sh[0]; #number of examples.
    n = sh[1]; #number of features.
    
    total_layers = len(netConfig);
    wt_layers= total_layers - 1;

    
    labels = np.zeros((m,1));
    
    X0_test = np.ones((m,1))
    X_new = np.hstack((X_test,X0_test))
    X_new = np.transpose(X_new);
    
    act_layer = {};
    act_layer_bias = {};
    act_layer_bias["hidden_output_bias0"] = X_new;
       
    for i in range(wt_layers):
        prev_wt = wt["wt_hidden"+str(i)];
        prev_ho =  act_layer_bias["hidden_output_bias"+str(i)];
        hidden_input = np.matmul(prev_wt,prev_ho);
            
        if(i+1 < wt_layers):
                
            if(act == "sigmoid"):
                hidden_output1 = sigmoid(hidden_input);
                
            elif(act == "swish"):
                hidden_output1 = sigmoid(hidden_input);
                act_layer["sigmoid_output"+str(i+1)] = hidden_output1;
                hidden_output1 = swish(hidden_input,hidden_output1);
                
            elif(act == "relu"):
                act_layer["hidden_input"+str(i+1)] = hidden_input;
                hidden_output1 = relu(hidden_input);
    
            hidden_output = np.vstack((hidden_output1,np.ones((1,m)))); #p+1Xm
            act_layer_bias["hidden_output_bias" + str(i+1)] = hidden_output;
            
        if(i+1 == wt_layers):
            hidden_output1 = softmax(hidden_input);
            act_layer["hidden_output"+str(i+1)] = hidden_output1;
            if(netConfig[wt_layers] == 1):
                for j in range(m):
                    if(hidden_output1[0,j] >=0.5):
                        labels[j,0] = 1;
                    else:
                        labels[j,0] = 0;
            elif(netConfig[wt_layers] > 1): 
                for j in range(m):
                    hidden_output1 = np.round(hidden_output1,2);
                    labels[j,0] = np.argmax((hidden_output1[:,j]));
                
    return labels;


def EvaluateAcc(Y_predict, Y):
    
    err  = 0.0;
    sh = Y.shape;
    for i in range (sh[0]):
        if Y_predict[i] != Y[i]:
            err = err+1.0;
    
    percent = ((sh[0]-err)/sh[0])*100.0;
    
    
    return percent;





from IPython.display import display
from PIL import Image




def MNIST_train(dirNameTrain,dirNameTest,config):
        labels = [0,1,2,3, 4,5,6,7,8,9];
        #labels = [0,1];
        X_raw = []
        Y = []
        for label in labels:
            dirName = '../data/MNIST/'+str(label);
            imgList = os.listdir(dirName);
            for img in imgList:
                    X_raw.append(scipy.ndimage.imread(os.path.join(dirName,img)))
                    Y.append(label);

        X_rawTest = []
        Y_Test = []
        for label in labels:
            dirName = dirNameTest + '/' + str(label)
            imgList = os.listdir(dirName);
            for img in imgList:
                    X_rawTest.append(scipy.ndimage.imread(os.path.join(dirName,img)))
                    Y_Test.append(label);

        X = []
        X_Test=[]

        for x in X_raw:
            X.append(x.flatten());
        X = np.array(X);
        Y = np.array(Y).reshape((X.shape[0],1));
        YX = np.concatenate((Y,X),axis=1);

        for x in X_rawTest:
            X_Test.append(x.flatten());
        X_Test = np.array(X_Test);
        Y_Test = np.array(Y_Test).reshape((X_Test.shape[0],1));
        YX_Test = np.concatenate((Y_Test,X_Test),axis=1);

        #YX_train, YX_test = train_test_split(YX, train_size = 0.7);

        X_train = YX[:,1:];
        Y_train = YX[:,0].reshape((YX.shape[0],1));

        X_test = YX_Test[:,1:];
        Y_test = YX_Test[:,0].reshape((YX_Test.shape[0],1));


        X_test = (X_test)/255
        X_train =(X_train)/255
        #print(X_train[0]);
        #print(X_test.shape)
        #print(X_train.shape)

        sh = X_train.shape;
        n = sh[1]; #number of features.






        batch_size = 64;
        config.insert(len(config),10)
        config.insert(0,n)
        
        
        netconfig = np.array(config);
        #print(netconfig)
        params =  trainNeuralNet_unified(X_train, Y_train,netconfig,batch_size,0.03,10000,"sigmoid");

        Y_prediction =  predictNeuralNet_unified(X_test,netconfig,params,"sigmoid");
        acc = EvaluateAcc(Y_prediction, Y_test);
       
        print("Accuracy:",acc)
        print("F1score(macro): ",f1_score( Y_test, Y_prediction,average='macro')*100)
        print("F1Score(micro) :",f1_score( Y_test, Y_prediction,average='micro')*100)





def MNIST_test(dirNameTest):
        labels = [0,1,2,3, 4,5,6,7,8,9];
        #labels = [0,1];
        X_raw = [];
        Y = []
        for label in labels:
            dirName = dirNameTest+'/'+str(label);
            imgList = os.listdir(dirName);
            for img in imgList:
                X_raw.append(scipy.ndimage.imread(os.path.join(dirName,img)));
                Y.append(label);

        X = [];
        for x in X_raw:
            X.append(x.flatten());
        X = np.array(X);
        Y = np.array(Y).reshape((X.shape[0],1));
        YX = np.concatenate((Y,X),axis=1);

        

        

        X_test = YX[:,1:];
        Y_test = YX[:,0].reshape((YX.shape[0],1));


        X_test = (X_test)/255;
       
        
       
        
        
        f = open("MNISTWeights", "rb")
        params = pickle.load(f)
        f.close()
        netconfig = np.array([784,30,30,10]);
        Y_prediction =  predictNeuralNet_unified(X_test,netconfig,params,"sigmoid");
        acc = EvaluateAcc(Y_prediction, Y_test);
        
        print("Accuracy:",acc)
        print("F1score(macro): ",f1_score( Y_test, Y_prediction,average='macro')*100)
        print("F1Score(micro) :",f1_score( Y_test, Y_prediction,average='micro')*100)







def CatDog_train(dirNameTrain,dirNameTest,config):
        labels = [0, 1]

        X_raw = []
        Y = []
        animals = ['cat','dog']
        for label in range(2):
            dirName = '../data/Cat-Dog/'+str(animals[label])
            #print(dirName)
            imgList = os.listdir(dirName)
            for img in imgList:
                X_raw.append(plt.imread(os.path.join(dirName,img)))
                Y.append(label)


        X_rawTest = []
        Y_Test = []
        for label in labels:
            dirName = dirNameTest + '/' + str(animals[label])
            imgList = os.listdir(dirName);
            for img in imgList:
                    X_rawTest.append(scipy.ndimage.imread(os.path.join(dirName,img)))
                    Y_Test.append(label);

        X = []
        X_Test=[]

        for x in X_raw:
            X.append(x.flatten());
        X = np.array(X);
        Y = np.array(Y).reshape((X.shape[0],1));
        YX = np.concatenate((Y,X),axis=1);

        for x in X_rawTest:
            X_Test.append(x.flatten());
        X_Test = np.array(X_Test);
        Y_Test = np.array(Y_Test).reshape((X_Test.shape[0],1));
        YX_Test = np.concatenate((Y_Test,X_Test),axis=1);

        #YX_train, YX_test = train_test_split(YX, train_size = 0.7);

        X_train = YX[:,1:];
        Y_train = YX[:,0].reshape((YX.shape[0],1));

        X_test = YX_Test[:,1:];
        Y_test = YX_Test[:,0].reshape((YX_Test.shape[0],1));


        #X_test = (X_test)/256
        #X_train =(X_train)/256
        #print(X_train[0]);
        #print(X_test.shape)
        #print(X_train.shape)

        sh = X_train.shape;
        n = sh[1]; #number of features.
        batch_size = 64;
        config.insert(len(config),2)
        config.insert(0,n)
        
        
        netconfig = np.array(config);
        #print(netconfig)
        params =  trainNeuralNet_unified(X_train, Y_train,netconfig,batch_size,0.03,100,"sigmoid");

        Y_prediction =  predictNeuralNet_unified(X_test,netconfig,params,"sigmoid");
        acc = EvaluateAcc(Y_prediction, Y_test);
        print("Accuracy: ",acc);
        print("F1score(macro): ",f1_score( Y_test, Y_prediction,average='macro')*100)
        print("F1Score(micro): ",f1_score( Y_test, Y_prediction,average='micro')*100)




def CatDog_test(dirNameTest):
        labels = [0, 1]

        X_raw = []
        Y = []
        animals = ['cat','dog']
        for label in range(2):
            dirName =  dirNameTest+'/'+str(animals[label])
            #print(dirName)
            imgList = os.listdir(dirName)
            for img in imgList:
                X_raw.append(plt.imread(os.path.join(dirName,img)))
                Y.append(label)


        X = [];
        for x in X_raw:
            X.append(x.flatten());
        X = np.array(X);
        Y = np.array(Y).reshape((X.shape[0],1));
        YX = np.concatenate((Y,X),axis=1);
        X_test = YX[:,1:];
        Y_test = YX[:,0].reshape((YX.shape[0],1));
        
        f = open("CatDogWeights", "rb")
        params = pickle.load(f)
        f.close()
        netconfig = [120000,1024,64,2]
        Y_prediction =  predictNeuralNet_unified(X_test,netconfig,params,"sigmoid");
        acc = EvaluateAcc(Y_prediction, Y_test);
        print("Accuracy:",acc)
        print("F1score(macro): ",f1_score( Y_test, Y_prediction,average='macro')*100)
        print("F1Score(micro) :",f1_score( Y_test, Y_prediction,average='micro')*100)


# In[29]:

#MNIST_train('../data/MNIST','../data/MNIST',[30,30,30])
#MNIST_test('../data/MNIST')
#CatDog_train('../data/Cat-Dog','../data/Cat-Dog',[30])
#CatDog_test('../data/Cat-Dog')


# In[ ]:




