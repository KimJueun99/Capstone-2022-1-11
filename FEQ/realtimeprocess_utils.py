import pandas as pd
import h5py
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import random
import csv
from datetime import datetime
import time
import random
from itertools import count
import matplotlib.pyplot as plta
from matplotlib.animation import FuncAnimation
from pandas.core.indexes import interval
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from FEQtransformer.core.FEQT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization

#from FEQtransformer.core.predictor import predictor

def make_stream(dataset):
    '''
    input: hdf5 dataset
    output: obspy stream

    '''
    data = np.array(dataset)

    tr_E = obspy.Trace(data=data[:, 0])
    tr_E.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_E.stats.delta = 0.01
    tr_E.stats.channel = dataset.attrs['receiver_type']+'E'
    tr_E.stats.station = dataset.attrs['receiver_code']
    tr_E.stats.network = dataset.attrs['network_code']

    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_N.stats.delta = 0.01
    tr_N.stats.channel = dataset.attrs['receiver_type']+'N'
    tr_N.stats.station = dataset.attrs['receiver_code']
    tr_N.stats.network = dataset.attrs['network_code']

    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = dataset.attrs['receiver_type']+'Z'
    tr_Z.stats.station = dataset.attrs['receiver_code']
    tr_Z.stats.network = dataset.attrs['network_code']

    stream = obspy.Stream([tr_E, tr_N, tr_Z])

    return stream


def make_plot(tr, title='', ylab=''):
    '''
    input: trace
    
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax.axes(ylim=(-620,620))
    ax.plot(tr.times("matplotlib"), tr.data, "k-")
    ax.xaxis_date()
    fig.autofmt_xdate()
    
    

def readfile(csv_file_name,hdf5_file_name):
    csv_file = csv_file_name
    df = pd.read_csv(csv_file,index_col=0)
    ev_list = df['trace_name'].to_list()

    file_name =  hdf5_file_name
    dtfl = h5py.File(file_name,'r')

    time_data = []
    raw_data = []
    for evi in  ev_list:
        data = dtfl.get('data/'+evi)
        st = make_stream(data)
        tr = st[2]
        time_data.append(tr.times("matplotlib"))
        raw_data.append(tr.data)

    dtfl.close()

    return time_data, raw_data


def get_model(input_model):
    model1 = load_model(input_model, custom_objects={'SeqSelfAttention': SeqSelfAttention, 
                                                         'FeedForward': FeedForward,
                                                         'LayerNormalization': LayerNormalization, 
                                                         'f1': f1                                                                            
                                                         })
                                                        
    model1.compile(loss = ['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
                  loss_weights =  [0.03, 0.40, 0.58],           
                  optimizer = Adam(lr = 0.001),
                  metrics = [f1])
    
    return model1
                                                        
