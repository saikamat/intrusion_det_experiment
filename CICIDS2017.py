import numpy as np
import os
import csv
from os.path import join as os_join

def get_filenames(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isfile(os.path.join(a_dir, name))]


def read_csv_header(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        return header    


def read_csv(filename, sampling_rate=100):
    with open(filename) as csv_file:
        print('Reading {} ...'.format(filename))
        reader = csv.reader(csv_file)
        header = next(reader)
        data = [row for row in reader]        
        print('#{} rows read'.format(len(data)))
        N = len(data)
        sample_size = N*sampling_rate//100
        indices = np.random.randint(0,N, sample_size)   
        sampled_data = [data[i] for i in indices]  
    return sampled_data


def read_data(dataroot, sampling_rate=10, seed=0):
    np.random.seed(seed)    
    filenames = get_filenames(dataroot)
    data = []
    for filename in filenames:
        data_part = read_csv(os_join(dataroot,filename),sampling_rate)
        data+=data_part
    return data