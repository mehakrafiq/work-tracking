from optparse import OptionParser

parser = OptionParser()



parser.add_option("--model_name", "--fp", dest="Model_Name", help="Write the model name you want to train.For example:MLP,LSTM,CONVLSTM,CNNLSTM")
parser.add_option("--data_path","--dp",  dest="Data_Path", help="Path to training data features")
parser.add_option("--tags_path", "--tp", dest="Tags_Path", help="Path to training data labels or tags")

parser.add_option("--verbose", "--vr", dest="Verbose", help="Display training progress=1, not display=0",default=1)
parser.add_option("--epochs", "--ep", dest="Epochs", help="No. of epochs for model training",default=1000)
parser.add_option("--batch_size", "--bs", dest="Batch_size", help="Batch size for model training",default=256)

parser.add_option("--Model_Evaluation", "--me", dest="Model_evaluation", help="Want to save training loss acc graph, and test score",default=True)
parser.add_option("--Frozen_file", "--fr", dest="frozen_file", help="Want to save pb and pbtxt frozen file",default=False)



(options, args) = parser.parse_args()

print ('Please Wait Program running ...')

model_name=options.Model_Name
path_data=options.Data_Path
path_tags=options.Tags_Path
verbose=options.Verbose
epochs=options.Epochs
batch_size=options.Batch_size
Model_Evaluation=options.Model_evaluation
Generate_frozen_file=options.frozen_file

from active_utils.utils import Dataload
from active_utils.models import lstm_model, Cnn_lstm_model, ConvLstm, Mlp,get_callbacks_list

trainX,testX,valX,trainy,testy,valy=Dataload(path_data,path_tags)

from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import dump
scalers = {}
for i in range(trainX.shape[2]):
    scalers[i] = StandardScaler()
    trainX[:,:,i] = scalers[i].fit_transform(trainX[:,:,i]) 

    
for i in range(testX.shape[2]):
    testX[:,:,i] = scalers[i].transform(testX[:,:,i]) 


for i in range(valX.shape[2]):
    valX[:,:,i] = scalers[i].transform(valX[:,:,i]) 

dump(scalers, 'MODELS/'+model_name+'_std_scaler.bin', compress=True);

if model_name=='LSTM':
    print ('Loading LSTM model')
    model=lstm_model(trainX, trainy)

if model_name=='CONVLSTM':
    print ('Loading CONVLSTM model')
    n_steps, n_length = 6, 50
    model=ConvLstm(trainX, trainy, n_steps, n_length)
    
    # reshape data into time steps of sub-sequences
    
    n_features=trainX.shape[2]
    trainX = trainX.reshape((trainX.shape[0], n_steps, 1,n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps,1, n_length, n_features))
    valX = valX.reshape((valX.shape[0], n_steps,1, n_length, n_features))

if model_name=='CNNLSTM':
    print ('Loading CNNLSTM model')
    n_steps, n_length = 6, 50
    model=Cnn_lstm_model(trainX, trainy,n_steps, n_length)
    
    # reshape data into time steps of sub-sequences
    
    n_features=trainX.shape[2]
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
    valX = valX.reshape((valX.shape[0], n_steps, n_length, n_features))
    

if model_name=='MLP':
    print ('Loading MLP model')
    model=Mlp(trainX, trainy)
    
    trainX=trainX.reshape(trainX.shape[0],trainX.shape[1]*trainX.shape[2])
    testX=testX.reshape(testX.shape[0],testX.shape[1]*testX.shape[2])
    valX=valX.reshape(valX.shape[0],valX.shape[1]*valX.shape[2])

model.summary()
from keras.utils import plot_model
plot_model(model, to_file='MODELS/'+model_name+'.png',show_shapes=True,show_layer_names=False)

print ('Model Training Starting:')
hist=model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose,validation_data=[valX,valy],callbacks=get_callbacks_list(model_name))


if Model_Evaluation:
    from active_utils.Performance_Saver import model_Evaluate, save_training_images

    save_training_images(model_name,hist)

    model_Evaluate(model,trainX, trainy,testX, testy,valX, valy,batch_size,model_name,path_data, epochs)

## Model Saving tensorflow format

from keras.models import load_model
import tensorflow as tf
import keras
from active_utils.models import freeze_session

if Generate_frozen_file:
    model=load_model('MODELS/'+model_name+'.h5')

    frozen_graph = freeze_session(keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
    #tf.train.write_graph(frozen_graph, 'MODELS/', model_name+'_frozen_save.pbtxt', as_text=True)
    tf.train.write_graph(frozen_graph, 'MODELS/', model_name+'_frozen_save.pb', as_text=False)
    print ('Frozen files saved in MODELS directory')