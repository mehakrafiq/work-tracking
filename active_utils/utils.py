from sklearn.model_selection import train_test_split
import numpy as np

def Dataload(path_data,path_tags):
    tags=np.load(path_tags)
    Data_total=np.load(path_data)

    Division=int(tags.shape[0]/3.5)

    Data_total=Data_total[:Division]
    tags=tags[:Division]

    trainX,testX,trainy,testy=train_test_split(Data_total,tags,test_size=0.2,random_state=3)

    testX,valX,testy,valy=train_test_split(testX,testy,test_size=0.5,random_state=3)
    return trainX,testX,valX,trainy,testy,valy