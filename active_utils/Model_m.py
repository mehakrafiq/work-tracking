import numpy as np
from sklearn.externals.joblib import load

def Model_Result(model,data,Standard_scalar_path):
    X=data.iloc[:,:].values
    total_data=[]
    for i in range(len(X)):
        for j in range(X[:,2][i]):
            total_data.append(X[i])

    total_data_new=np.array(total_data)
    total_data=total_data_new[:,[2,4,5,6,8]]

    total_data=total_data.astype('float64')

    indexs=list(range(0,total_data.shape[0],300))

    n_steps,n_length  = 6, 50
    data_2_test=total_data[indexs[0]:indexs[-1]].reshape(len(indexs)-1,300,5)


    scalers=load(Standard_scalar_path)

    for i in range(data_2_test.shape[2]):
        data_2_test[:,:,i] = scalers[i].transform(data_2_test[:,:,i]) 
    n_features=data_2_test.shape[2]
    data_2_test = data_2_test.reshape((data_2_test.shape[0], n_steps, n_length, n_features))
    data_2_test.shape



    predicted=model.predict(data_2_test,batch_size=256)

    b = np.zeros_like(predicted)
    b[np.arange(len(predicted)), predicted.argmax(1)] = 1

    Test_Class=predicted.argmax(axis=1)

    class_names=['Partially Working','Not Working','Working']

    lines=np.array([Test_Class]*300).T.reshape(-1,)
    lines[lines==1]=10
    lines[lines==2]=5


    TOTAL_DATA=np.hstack((total_data_new[:lines.shape[0],:],lines.reshape(lines.shape[0],1)))

    TOTAL_DATA_LIST=TOTAL_DATA.tolist()
    b = list()
    for sublist in TOTAL_DATA_LIST:
        if sublist not in b:
            b.append(sublist)
    new_array_data=np.array(b)

    a=new_array_data[:,1]
    index_sets = [np.argwhere(i==a) for i in np.unique(a)]

    repeated_elements=[]
    for i in range(len(index_sets)):
        if len(index_sets[i])>1:
            repeated_elements.append(index_sets[i][0][0])

    Data_to_save=np.delete(new_array_data,repeated_elements, axis=0)
    return Data_to_save


def summary_saving(Time_Saving,File_path):
    from datetime import timedelta
    from datetime import datetime

    time_saving_array=[]

    time_saving_array.append(['File','Activity Start Time','Activity End Time','total Activity Time','Inside Area Total Time','Inside Area Working Time','Percentage Area Inside Time','Percentage Area Working Time'])
    inside_medium_time_save=[]
    inside_all_time_save=[]
    for i in range(len(Time_Saving)):
        if 'Medium start inside:' in Time_Saving[i][0]:

            s1 = Time_Saving[i][1] #'10:33:26'
            s2 = Time_Saving[i+1][1] #'11:33:26' # for example
            FMT = '%H:%M:%S'
            inside_m_time = (datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)).total_seconds()

    #             print (outside_time)
            inside_medium_time_save.append(inside_m_time)

        if 'start inside:' in Time_Saving[i][0]:

            s1 = Time_Saving[i][1] #'10:33:26'
            s2 = Time_Saving[i+1][1] #'11:33:26' # for example
            FMT = '%H:%M:%S'
            inside_all_time = (datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)).total_seconds()

    #             print (outside_time)
            inside_all_time_save.append(inside_all_time)
    
    s1 = Time_Saving[0][1] #'10:33:26'
    s2 = Time_Saving[-1][1] #'11:33:26' # for example
    FMT = '%H:%M:%S'
    total_activity_time = (datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)).total_seconds()

    time_saving_array0=[]
    time_saving_array0.append(File_path)
    time_saving_array0.append(str(Time_Saving[0][1]))
    time_saving_array0.append(str(Time_Saving[-1][1]))

    time_saving_array0.append(str(timedelta(seconds=total_activity_time)))

    time_saving_array0.append(str(timedelta(seconds=sum(inside_all_time_save))))
    time_saving_array0.append(str(timedelta(seconds=sum(inside_medium_time_save))))
    percentage_area_time=(sum(inside_all_time_save)/total_activity_time)*100
    if len(inside_all_time_save)>0:    
        percentage_area_medium_time=(sum(inside_medium_time_save)/sum(inside_all_time_save))*100
    else:
        print ('Kindly Choose the right Area Map, or there is no point inside the choosen Area')
        percentage_area_medium_time=0
    time_saving_array0.append(percentage_area_time)
    time_saving_array0.append(percentage_area_medium_time)
    time_saving_array.append(time_saving_array0)


    # wb = Workbook(write_only=True)
    # ws = wb.create_sheet()

    # # now we'll fill it with 100 rows x 200 columns
    # for irow in time_saving_array:
    #     ws.append(irow)  
    # # save the file
    # wb.save('Results_Time.xlsx')

    with open('Result/summary.txt', 'w') as f:
        for ind in range(len(time_saving_array[0])):
                f.write("%s\n" % (time_saving_array[0][ind]+' :'+str(time_saving_array[1][ind])))
                   
    