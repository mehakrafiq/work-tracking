from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import csv
import os

def save_training_images(model_name,hist):
    plt.figure(figsize=[10,8])
    plt.plot(hist.history['val_loss'],label='val_loss');
    plt.plot(hist.history['loss'],label='train_loss');
    plt.xlabel('No. of Epochs')
    plt.ylabel('Loss')
    plt.title('Loss/Epochs')
    plt.legend()
    plt.savefig('Images/'+model_name+'_loss.png')

    plt.figure(figsize=[10,8])
    plt.plot(hist.history['val_acc'],label='val_acc');
    plt.plot(hist.history['acc'],label='train_acc');
    plt.xlabel('No. of Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy/Epochs')
    plt.legend()
    plt.savefig('Images/'+model_name+'_acc.png');
    print ('Training accuracy and loss images saved in Images directory')



def model_Evaluate(model,trainX, trainy,testX, testy,valX, valy,batch_size,model_name,path_data, epochs):


    if not os.path.exists('Performance_comparison.csv'):
        fields=['Model Name','Data Path','No.of Epochs', 'Batch Size','Training_data Accuracy','Testing_data Accuracy',
                'Validation_data Aaccuracy','Training_data Loss','Testing_data Loss','Validation_data Loss',
                'Partially_working correct prediction','Not_working correct prediction','Working correct prediction','Wrong_prediction']
        with open(r'Performance_comparison.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            
    labels=["Partially Working", "Not Working", "Working"]
    Loss_tr, accuracy_tr = model.evaluate(trainX, trainy, batch_size=batch_size, verbose=0)
    Loss_ts, accuracy_ts = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    Loss_vl, accuracy_vl = model.evaluate(valX, valy, batch_size=batch_size, verbose=0)

    y_pred=model.predict(testX)
    b = np.zeros_like(y_pred)
    b[np.arange(len(y_pred)), y_pred.argmax(1)] = 1

    #columns are predicted, rows are truth
    predicted = y_pred.argmax(axis=1)
    y_index = testy.argmax(axis=1)
    confusion_matrix = pd.crosstab(pd.Series(y_index), pd.Series(predicted))
    confusion_matrix.index = [labels[i] for i in confusion_matrix.index]
    confusion_matrix.columns = [labels[i] for i in confusion_matrix.columns]
    confusion_matrix.reindex(columns=[l for l in labels], fill_value=0);



    correct_partially_working=confusion_matrix.values[0,0]
    correct_not_working=confusion_matrix.values[1,1]
    correct_working=confusion_matrix.values[2,2]
    Wrong_prediction=np.sum(confusion_matrix.values)-correct_partially_working-correct_not_working-correct_working


    fields=[model_name,path_data, epochs, batch_size,accuracy_tr,accuracy_ts,accuracy_vl,Loss_tr,Loss_ts,Loss_vl,correct_partially_working,correct_not_working,correct_working,Wrong_prediction]

    with open(r'Performance_comparison.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    print ('Model Performance on Test data writed in Performance_comparison.csv')
