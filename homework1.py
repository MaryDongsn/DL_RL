import torch
import tensorflow as tf
import numpy as np
import random as rd
from itertools import count
from torch.autograd import Variable
from torch import nn



POLY_DEGREE = 10;
THETA= torch.zeros(POLY_DEGREE+1,1)
DATASIZE = 200;
TEST_SIZE = 1000;

print(THETA)


def getData(data_size,sigma):
    x = torch.empty(data_size, ).uniform_(0, 1).type(torch.FloatTensor)
    f = torch.cos(2 * np.pi * x)
    Z = torch.normal(0, sigma ** 2, size=(data_size,))
    y = f + Z
    return Variable(x), Variable(y)


def make_data_features(data_size,sigma,poly_degree):
    data = getData(data_size,sigma)
    x = data[0]
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(0, poly_degree+1)], 1), tf.reshape(tf.transpose(data[1]),[data_size,1])


def hypothesis(x,theta):
    return tf.matmul(x, theta)


def getMSE(y_predict,y, data_size):
    mean_square_err = tf.reduce_sum((y_predict - y) ** 2) / data_size
    return mean_square_err


def GradientDescent(x,y, y_predict,data_size,theta,learning_rate):
    learned_theta = tf.add(theta, learning_rate * 2 * tf.matmul(tf.transpose(x), (y - y_predict)) / data_size)

    return learned_theta

# gd_test = GradientDescent(x_featured,THETA,tf.reshape(tf.transpose(test_x[1]),[DATASIZE,1]),0.04,DATASIZE)

# print(gd_test)
# print(tf.transpose(gd_test))

def fitData(sigma, epoch, theta,poly_degree, learning_rate,training_size,testing_size):
    current_theta = theta
    k = 0
    training_data = make_data_features(training_size,sigma,poly_degree)
    x_training = training_data[0]
    y_training = training_data[1]
    #print("y_training", y_training)
    testing_data = make_data_features(testing_size,sigma,poly_degree)
    x_testing = testing_data[0]
    y_testing = testing_data[1]
    #print("y_testing",y_testing)

    while k<epoch:
        y_predict = hypothesis(x_training, current_theta)
        #print("y_predict", y_predict)
        y_test_predict = hypothesis(x_testing, current_theta)
        #print("y_test_predict", y_test_predict)
        training_lost = getMSE(y_predict, y_training, training_size)
        current_theta = GradientDescent(x_training,y_training,y_predict,training_size,current_theta,learning_rate)
        test_lost = getMSE(y_test_predict, y_testing,testing_size)
        k += 1
        #print("+++++++lost:", training_lost)
        #print("______out lost", test_lost)
    E_in = training_lost
    E_out = test_lost
    return current_theta, E_in, E_out

fitDataTest = fitData(0.01,1000,THETA,POLY_DEGREE,0.5,DATASIZE,TEST_SIZE)
print(fitDataTest)




def experiment(sigma, epoch, theta,poly_degree, learning_rate,training_size,testing_size,trails):
    k = 0
    Ein_total = 0
    Eout_total = 0
    theta_degree = poly_degree+1
    theta_total = torch.ones(theta_degree,1)
    while k < trails:
        fit_data = fitData(sigma, epoch, theta,poly_degree, learning_rate,training_size,testing_size)
        print("fit data test", fit_data)
        fit_theta = fit_data[0]

        training_lost = fit_data[1]
        print("+++++fit data test, Ein", fit_data[1])
        test_lost = fit_data[2]

        Ein_total += training_lost

        print("00000000+++++fit data test, total Ein", Ein_total)
        Eout_total += test_lost
        theta_total += fit_theta
        k+=1
    Ein_avg = Ein_total/trails
    Eout_avg = Eout_total/trails
    theta_avg = theta_total/trails

    testing_data = make_data_features(testing_size,sigma,poly_degree)
    x_testing = testing_data[0]
    y_testing = testing_data[1]
    y_test_predict = hypothesis(x_testing, theta_avg)
    test_lost = getMSE(y_test_predict, y_testing, testing_size)



    return Ein_avg, Eout_avg,theta_avg, test_lost
experiment_test = experiment(0.01,10000,THETA,POLY_DEGREE,0.25,DATASIZE,TEST_SIZE,50)
print(experiment_test)