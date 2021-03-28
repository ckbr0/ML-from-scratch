import gzip

import numpy as np

def load_training_data():
    num_images = 60000

    training_inputs_file = gzip.open('../data/train-images-idx3-ubyte.gz', 'rb')
    training_inputs_file.seek(4*4)
    training_inputs_buffer = np.frombuffer(training_inputs_file.read(), dtype=np.uint8)*(1/255)
    training_inputs = np.split(training_inputs_buffer, num_images)

    training_results_file = gzip.open('../data/train-labels-idx1-ubyte.gz', 'rb')
    training_results_file.seek(2*4)
    training_results_buffer = np.frombuffer(training_results_file.read(), dtype=np.uint8)
    training_results = [vectorized_result(i) for i in np.split(training_results_buffer, num_images)]

    return list(zip(training_inputs, training_results))

def load_test_data():
    num_images = 10000

    test_inputs_file = gzip.open('../data/t10k-images-idx3-ubyte.gz', 'rb')
    test_inputs_file.seek(4*4)
    test_inputs_buffer = np.frombuffer(test_inputs_file.read(), dtype=np.uint8)*(1/255)
    test_inputs = np.split(test_inputs_buffer, num_images)

    test_results_file = gzip.open('../data/t10k-labels-idx1-ubyte.gz', 'rb')
    test_results_file.seek(2*4)
    test_results_buffer = np.frombuffer(test_results_file.read(), dtype=np.uint8)
    test_results = [i for i in np.split(test_results_buffer, num_images)]

    return list(zip(test_inputs, test_results))

def load_data():
    training_data = load_training_data()
    test_data = load_test_data()
    
    return training_data, test_data

def vectorized_result(i):
    v = np.zeros(10)
    v[i] = 1.0
    
    return v
