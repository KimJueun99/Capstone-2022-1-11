from realtimeprocess_utils import readfile,get_model
import matplotlib
matplotlib.use('TKAgg')## 2) Matplotlib Test
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import csv
from datetime import datetime
from FEQtransformer.core.predictor3 import predictor

datetime_format = "%Y-%m-%d %H:%M:%S.%f"


#1)load model
input_model_name = 'home/EQT/5_FEQ_test_trainer_outputs/final_model.h5'
model = get_model(input_model_name)
print('complete load model!')

#2)read data
csvfile_name = 'home/EQT/realtime_process/datas/new_4data/new_4data.csv'
hdf5file_name ='home/EQT/realtime_process/datas/new_4data/new_4data.hdf5'

time_data,raw_data =readfile(csvfile_name,hdf5file_name)

p_arrival_time = 'Nan'
s_arrival_time = 'Nan'
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.xaxis_date()   
plt.ylabel('counts')
plt.title('Raw Data')
plt.ylim([-620,620])
input_dir_name = 'home/EQT/realtime_process/datas/4data/data'
output_dir_name = 'home/EQT/realtime_process/test_predictor/test25'
for i in range(len(raw_data)):
    x1 = time_data[i].tolist()
    y1 = raw_data[i].tolist()
    
    plt.plot(x1,y1)
    
    p,s,delta = predictor(input_dir=input_dir_name+str(i+1),
            input_model=model,
            output_dir=output_dir_name,
            detection_threshold=0.3,                
            P_threshold=0.1,
            S_threshold=0.1, 
            number_of_plots=10,
            number_of_cpus=4)
    plt.pause(delta)
    print(p,s)
    #plt.pause(delta)
    if(p!='Nan' and s!='Nan'):
        p_arrival_time = datetime.strptime(p,datetime_format)
        s_arrival_time = datetime.strptime(s,datetime_format)
        print('[p_arrivaltime, s_arrivaltime] : [',p_arrival_time,', ',s_arrival_time,']')
        
        plt.plot([p_arrival_time,p_arrival_time],[-650,650],'k--',linewidth='1')
        plt.plot([s_arrival_time,s_arrival_time],[-650,650],'k--',linewidth='1')
    
    
    
#    plt.pause(1)


plt.show()