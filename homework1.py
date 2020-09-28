import torch
import tensorflow as tf
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns


SIGMA_LIST = [0.01, 0.1,1]
TRAINING_SIZE = [2,5,10,20,50,100,200]
DEGREE_LIST = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

# Values
TRIALS = 50
LEARNING_RATE = 0.25
TEST_SIZE = 1000


# Values used for plotting purposes
POLY_DEGREE = 10
THETA = torch.zeros(POLY_DEGREE+1,1)
SIGMA = 0.01


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


def GradientDescent(x,y, y_predict,data_size,theta,learning_rate, regularization):
    reg_lambda = 0.001
    learned_theta = tf.add(theta, learning_rate * 2 * tf.matmul(tf.transpose(x), (y - y_predict)) / data_size)
    if regularization:
        learned_theta = tf.add(learned_theta, -2 * learning_rate * reg_lambda * theta)
    return learned_theta


def fitData(sigma, epoch, theta,poly_degree, learning_rate,training_size,testing_size,regularization,plot):
    current_theta = theta
    k = 0
    errors = []
    epoch_time = []
    errors = np.array(errors)
    epoch_time = np.array(epoch_time)

    training_data = make_data_features(training_size,sigma,poly_degree)
    x_training = training_data[0]
    y_training = training_data[1]

    testing_data = make_data_features(testing_size,sigma,poly_degree)
    x_testing = testing_data[0]
    y_testing = testing_data[1]

    while k < epoch:
        y_predict = hypothesis(x_training, current_theta)
        y_test_predict = hypothesis(x_testing, current_theta)
        training_lost = getMSE(y_predict, y_training, training_size)
        current_theta = GradientDescent(x_training,y_training,y_predict,training_size,current_theta,learning_rate,regularization)
        test_lost = getMSE(y_test_predict, y_testing,testing_size)
        k += 1
        if plot:
            errors = np.append(errors,test_lost)
            epoch_time = np.append(epoch_time,k)
            print("+++++error array", errors)
            print("&&&&&&&&&&&etime", epoch_time)
    if plot:
        plotmse(epoch_time,errors)

    E_in = training_lost
    E_out = test_lost
    return current_theta, E_in, E_out

def experiment(sigma, epoch, theta,poly_degree, learning_rate,training_size,testing_size,trials,regularization,plot):
    k = 0
    Ein_total = 0
    Eout_total = 0
    theta_degree = poly_degree+1
    theta_total = torch.ones(theta_degree,1)

    while k < trials:
        print('Trial: ', k+1)
        fit_data = fitData(sigma, epoch, theta,poly_degree, learning_rate,training_size,testing_size,regularization,plot)
        fit_theta = fit_data[0]
        training_lost = fit_data[1]
        test_lost = fit_data[2]
        Ein_total += training_lost
        Eout_total += test_lost
        theta_total += fit_theta
        k += 1

    Ein_avg = Ein_total/trials
    Eout_avg = Eout_total/trials
    theta_avg = theta_total/trials
    testing_data = make_data_features(testing_size,sigma,poly_degree)
    x_testing = testing_data[0]
    y_testing = testing_data[1]
    y_test_predict = hypothesis(x_testing, theta_avg)
    test_lost = getMSE(y_test_predict, y_testing, testing_size)

    return Ein_avg, Eout_avg, test_lost, theta_avg


def plotmse(times, errors):
    plt.plot(times, errors)
    plt.show()


# training data size


def experiment_size(N):
    errors_in = np.array([])
    errors_out = np.array([])
    errors_bias = np.array([])
    errors_in_reg = np.array([])
    errors_out_reg = np.array([])
    errors_bias_reg = np.array([])
    for n in N:
        print("n: ", n)
        experiment_N_reg = experiment(SIGMA, 10000, THETA, POLY_DEGREE, LEARNING_RATE, n, TEST_SIZE, TRIALS, True, False)
        experiment_N = experiment(SIGMA, 10000, THETA, POLY_DEGREE, LEARNING_RATE, n, TEST_SIZE, TRIALS, False, False)

        Ein_N_reg = experiment_N_reg[0]
        Eout_N_reg = experiment_N_reg[1]
        Ebias_N_reg = experiment_N_reg[2]

        Ein_N = experiment_N[0]
        Eout_N = experiment_N[1]
        Ebias_N = experiment_N[2]

        errors_in_reg = np.append(errors_in_reg, Ein_N_reg)
        errors_out_reg = np.append(errors_out_reg, Eout_N_reg)
        errors_bias_reg= np.append(errors_bias_reg, Ebias_N_reg)

        errors_in = np.append(errors_in, Ein_N)
        errors_out = np.append(errors_out, Eout_N)
        errors_bias= np.append(errors_bias, Ebias_N)

    fig, ax = plt.subplots()
    sns.lineplot(x=N, y=errors_in_reg, color='blue', label='E_in with regularization', ax=ax)
    sns.lineplot(x=N, y=errors_out_reg, color='red', label='E_out with regularization', ax=ax)
    sns.lineplot(x=N, y=errors_bias_reg, color='green', label='E_bias with regularization', ax=ax)

    sns.lineplot(x=N, y=errors_in, color='blue', label='E_in', ax=ax, linestyle='dashed')
    sns.lineplot(x=N, y=errors_out, color='red', label='E_out', ax=ax, linestyle='dashed')
    sns.lineplot(x=N, y=errors_bias, color='green', label='E_bias', ax=ax, linestyle='dashed')

    plt.xlabel('training sample size')
    plt.ylabel('MSE')
    plt.title('d = 10, sigma = 0.01')

    plt.savefig("Sample size 0.01 sigma experiment with "+str(TRIALS)+" trials.png")

def experiment_sigma(Sigma_list):
    errors_in = np.array([])
    errors_out = np.array([])
    errors_bias = np.array([])
    errors_in_reg = np.array([])
    errors_out_reg = np.array([])
    errors_bias_reg = np.array([])
    for sigma in Sigma_list:
        print("theta: ", sigma)
        experiment_N_reg = experiment(sigma, 2000, THETA, POLY_DEGREE, LEARNING_RATE, 200, TEST_SIZE, TRIALS, True, False)
        experiment_N = experiment(sigma, 2000, THETA, POLY_DEGREE, LEARNING_RATE, 200, TEST_SIZE, TRIALS, False, False)

        Ein_reg = experiment_N_reg[0]
        Eout_reg = experiment_N_reg[1]
        Ebias_reg = experiment_N_reg[2]

        Ein = experiment_N[0]
        Eout = experiment_N[1]
        Ebias= experiment_N[2]

        errors_in_reg = np.append(errors_in_reg, Ein_reg)
        errors_out_reg = np.append(errors_out_reg, Eout_reg)
        errors_bias_reg= np.append(errors_bias_reg, Ebias_reg)

        errors_in = np.append(errors_in, Ein)
        errors_out = np.append(errors_out, Eout)
        errors_bias = np.append(errors_bias, Ebias)

    fig, ax = plt.subplots()
    sns.lineplot(x=Sigma_list, y=errors_in_reg, color='blue', label='E_in with regularization', ax=ax)
    sns.lineplot(x=Sigma_list, y=errors_out_reg, color='red', label='E_out with regularization', ax=ax)
    sns.lineplot(x=Sigma_list, y=errors_bias_reg, color='green', label='E_bias with regularization', ax=ax)

    sns.lineplot(x=Sigma_list, y=errors_in, color='blue', label='E_in', ax=ax, linestyle='dashed')
    sns.lineplot(x=Sigma_list, y=errors_out, color='red', label='E_out', ax=ax, linestyle='dashed')
    sns.lineplot(x=Sigma_list, y=errors_bias, color='green', label='E_bias', ax=ax, linestyle='dashed')

    plt.xlabel('training sample size')
    plt.ylabel('MSE')
    plt.title('d = 10, N = 200')

    plt.savefig("Sigma experiment with "+str(TRIALS)+" trials.png")



def experiment_degree(degree_list):
    errors_in = np.array([])
    errors_out = np.array([])
    errors_bias = np.array([])
    errors_in_reg = np.array([])
    errors_out_reg = np.array([])
    errors_bias_reg = np.array([])
    for degree in degree_list:
        print("poly_degree: ", degree)
        theta = torch.zeros(degree+1,1)
        experiment_N_reg = experiment(SIGMA, 2000, theta, degree, LEARNING_RATE, 100, TEST_SIZE, TRIALS, True, False)
        experiment_N = experiment(SIGMA, 2000, theta, degree, LEARNING_RATE, 100, TEST_SIZE, TRIALS, False, False)

        Ein_reg = experiment_N_reg[0]
        Eout_reg = experiment_N_reg[1]
        Ebias_reg = experiment_N_reg[2]

        Ein = experiment_N[0]
        Eout = experiment_N[1]
        Ebias= experiment_N[2]

        errors_in_reg = np.append(errors_in_reg, Ein_reg)
        errors_out_reg = np.append(errors_out_reg, Eout_reg)
        errors_bias_reg= np.append(errors_bias_reg, Ebias_reg)

        errors_in = np.append(errors_in, Ein)
        errors_out = np.append(errors_out, Eout)
        errors_bias = np.append(errors_bias, Ebias)

    fig, ax = plt.subplots()
    sns.lineplot(x=degree_list, y=errors_in_reg, color='blue', label='E_in with regularization', ax=ax)
    sns.lineplot(x=degree_list, y=errors_out_reg, color='red', label='E_out with regularization', ax=ax)
    sns.lineplot(x=degree_list, y=errors_bias_reg, color='green', label='E_bias with regularization', ax=ax)

    sns.lineplot(x=degree_list, y=errors_in, color='blue', label='E_in', ax=ax, linestyle='dashed')
    sns.lineplot(x=degree_list, y=errors_out, color='red', label='E_out', ax=ax, linestyle='dashed')
    sns.lineplot(x=degree_list, y=errors_bias, color='green', label='E_bias', ax=ax, linestyle='dashed')

    plt.xlabel('polynomial degree')
    plt.ylabel('MSE')
    plt.title('theta = 0.01, N = 100')

    plt.savefig("Degree experiment with "+str(TRIALS)+" trials.png")

#experiment_size(TRAINING_SIZE)

#experiment_sigma(SIGMA_LIST)

experiment_degree(DEGREE_LIST)