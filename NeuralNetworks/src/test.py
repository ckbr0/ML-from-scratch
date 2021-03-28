import numpy as np
import cv2 as cv

from network_basic import Network
from mnist_db_loader import load_data
from activation_functions import sigmoid, sigmoid_prime

def cost_function_prime(x, y):
    return (x - y)

training_data, test_data = load_data()

"""arr1 = np.zeros((255, 255))
arr1 = cv.normalize(training_data[0][0].reshape((28,28)), arr1, 0, 255, cv.NORM_MINMAX)
cv.namedWindow('image1', cv.WINDOW_NORMAL)
cv.imshow('image1', arr1)
cv.waitKey(0)
cv.destroyWindow('image1')

arr2 = np.zeros((255, 255))
arr2 = cv.normalize(training_data_2[0][0].reshape((28,28)), arr2, 0, 255, cv.NORM_MINMAX)
cv.namedWindow('image2', cv.WINDOW_NORMAL)
cv.imshow('image2', arr2)
cv.waitKey(0)
cv.destroyWindow('image2')"""

network = Network([784, 32, 10], sigmoid, sigmoid_prime, cost_function_prime)
for i in range(10):
    network.stochastic_gradient_descent(training_data, 100, 3.0)
r = network.evaluate(test_data)
print(r)

network = Network([784, 32, 16, 10], sigmoid, sigmoid_prime, cost_function_prime)
for i in range(10):
    network.stochastic_gradient_descent(training_data, 100, 3.0)
r = network.evaluate(test_data)
print(r)

