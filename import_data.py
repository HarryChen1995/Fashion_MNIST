import os 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pickle

def creat_onehot(x):
    label=np.zeros((1,10),dtype=np.float32)
    label[:,int(x)]=1
    return label



def read_data():

    if not os.path.exists("/home/hanlin/Desktop/Recognize_Fashion/data.p"):
        train_data_frame=pd.read_csv("fashion-mnist_train.csv")
        test_data_frame=pd.read_csv("fashion-mnist_test.csv")
        print("finish reading training images.......")
        print("Training images shape{}".format(train_data_frame.shape))

        print("finish reading test images.......")
        print("Testing images shape{}".format(test_data_frame.shape))
        
        
        train_image=train_data_frame.iloc[:,1:]
        train_image=np.array(list(train_image.values)).reshape(-1,28,28,1)
        train_label=np.array(list(map(creat_onehot, train_data_frame['label'].values))).reshape(-1, 10)


        test_image=test_data_frame.iloc[:,1:]
        test_image=np.array(list(test_image.values)).reshape(-1,28,28,1)
        test_label=np.array(list(map(creat_onehot, test_data_frame['label'].values))).reshape(-1, 10)

        with open("/home/hanlin/Desktop/Recognize_Fashion/data.p","wb") as file:
            try:
                print("pickling.......")
                dataset={
                    'train_image':train_image,
                    'train_label':train_label,
                    'test_image':test_image,
                    'test_label':test_label

                }
                pickle.dump(dataset,file)
            except:

                print ("unable to pickling")



    with open("/home/hanlin/Desktop/Recognize_Fashion/data.p","rb") as file:

        data=pickle.load(file)
        train_image=data['train_image']
        train_label=data['train_label']
        test_image=data['test_image']
        test_label=data['test_label']
    

    return train_image,train_label,test_image,test_label