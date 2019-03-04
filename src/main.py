import nn
from sys import argv
import warnings
warnings.filterwarnings("ignore")
import argparse
import sys
import os




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', default= None)
    parser.add_argument('--test-data')
    parser.add_argument('--dataset')
    parser.add_argument('--configuration', nargs ="*")
    args = parser.parse_args()

    if(args.dataset == "MNIST"):
         #if os.path.exists(args.train_data):
        if args.train_data is not None:
                string = args.configuration
                string1 = string[0].strip('[')
                string2 = string[len(string)-1].strip(']')
                string[0] = string[0].strip('[')
                string[len(string)-1] = string[len(string)-1].strip(']')
                results = list(map(int, string))
                nn.MNIST_train(args.train_data,args.test_data,results)

        else:
                nn.MNIST_test(args.test_data)

    if(args.dataset == "Cat-Dog"):
         #if os.path.exists(args.train_data):
        if args.train_data is not None:
                string = args.configuration
                string1 = string[0].strip('[')
                string2 = string[len(string)-1].strip(']')
                string[0] = string[0].strip('[')
                string[len(string)-1] = string[len(string)-1].strip(']')
                results = list(map(int, string))
                nn.CatDog_train(args.train_data,args.test_data,results)

        else:
                nn.CatDog_test(args.test_data)



