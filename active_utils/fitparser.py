import fitparse
import pytz
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

#!conda install -c conda-forge folium=0.5.0 --yes
def fitparser(File_path):
    fitfile = fitparse.FitFile(File_path)
    datastr=[]
    datastr.append(['date','time','distance(m)','enhanced_speed(m/s)','speed(m/s)','heart_rate(bpm)','position_lat','position_long'])

    for g in range(len(fitfile.messages)):
        datastr0=[None]*8
        for i in fitfile.messages[g]:
            if i.name=='timestamp':
                date_time=i.value

                dt = pytz.timezone("Europe/London").localize(date_time)
                date=datetime.datetime.astimezone(dt).isoformat().split('+')[0].split('T')[0]
                time=datetime.datetime.astimezone(dt).isoformat().split('+')[0].split('T')[1]
                datastr0[0]=date
                datastr0[1]=time

            if i.name=='distance':
                datastr0[2]=i.value
            if i.name=='enhanced_speed':
                datastr0[3]=i.value
            if i.name=='speed':
                datastr0[4]=i.value
            if i.name=='heart_rate':
                datastr0[5]=i.value
            if i.name=='position_lat':
                if i.value is not None:
                    datastr0[6]=(i.value)*(180/(2**31))
            if i.name=='position_long':
                if i.value is not None:
                    datastr0[7]=(i.value)*(180/(2**31))

        if datastr0.count(None)<1:
            datastr.append(datastr0)
    data=pd.DataFrame(datastr)
    
    
    Full_data=data.iloc[:,:].values
    Full_data=np.array(Full_data)
    dates_of_work=np.unique(Full_data[:,0])

    date_index=0

    per_data_Data=Full_data[Full_data[:,0]==dates_of_work[date_index]]

    data=pd.DataFrame(per_data_Data,columns=['date','time','distance(m)','enhanced_speed(m/s)','speed(mm/s)','heart_rate(bpm)','position_lat','position_long'])
    #-----------------For image save-------------#
    plt.rcParams["figure.figsize"] = 30,12
    fig, ax1 = plt.subplots()


    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Heart Rate(bpm)', color='tab:red')
    ax1.plot(np.arange(len(data.iloc[:,5])), data.iloc[:,5], color='tab:red',linewidth=1)
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



    ax2.set_ylabel('Speed(KM/H)', color='tab:blue')  # we already handled the x-label with ax1
    ax2.plot(np.arange(len(data.iloc[:,3])), (data.iloc[:,3]/1000)*3600, color='tab:blue',linewidth=1)
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()
    plt.savefig('Result/graph.png')
    #------------------------------------------#
    
    data_To_Save=data.copy()

    data_To_Save=data_To_Save.drop(columns=['enhanced_speed(m/s)'])
    data_To_Save.to_csv('Result/Fit_File_Data.csv')


    X=data.iloc[:,[1,2,3,5]].values

    difference_of_distance=[]
    difference_of_distance.append(X[0][1])
    for i in range(1,len(X)):
        difference_of_distance.append(X[i][1]-X[i-1][1])

    idx = 3 
    data.insert(loc=idx, column='Distance Difference', value=difference_of_distance)

    from datetime import datetime as dtime

    difference_of_time=[]
    difference_of_time.append(0)
    for h in range(1,len(X)):

        FMT = '%H:%M:%S'
        tdelta = dtime.strptime(X[h][0], FMT) - dtime.strptime(X[h-1][0], FMT)
        difference_of_time.append(tdelta.seconds)

    idx = 2
    data.insert(loc=idx, column='Time Difference', value=difference_of_time)

    # Dropping 0 so that infinity does not appear.
    persecond_distance=[]
    X=data.iloc[:,[2,4]].values
    zero_value_index=np.where(X[:,0]==0)[0]
    initial=0
    for h in zero_value_index:
        data.drop(data.index[h-initial], inplace=True) # for droping values in dataframe
        initial+=1

    X=data.iloc[:,[2,4]].values
    for i in range(len(X)):
        persecond_distance.append(np.divide(X[i][1],X[i][0]))

    idx = 5 
    data.insert(loc=idx, column='Speed Calculated(m/s)', value=persecond_distance)

    X=data.iloc[:,:].values

    FMT = '%H:%M:%S'
    Total_Activity_seconds=(dtime.strptime(X[-1][1], FMT) - dtime.strptime(X[0][1], FMT)).seconds
    print ('Total_Activity_seconds: ',Total_Activity_seconds)
    
    return data