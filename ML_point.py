import numpy as np
import math
import matplotlib.pyplot as plt

alpha = 0.01 #task learning rate
beta = 0.01 #meta learning rate
K = 1 #number of gradient updates in task training
N = 10 #number of samples used for task training
M = 10 #number of samples used for task testing
J = 10 #number of different tasks to train on in each iteration

def sample_sin_task(N, task):
	input_points = np.random.uniform(-10., 10., size = N)
	A = task['amplitude']
	P = task['phase']
	output = A * np.sin(input_points + P)
	return input_points, output

def ML_point(theta, task): 
	#theta is weights of network
	input_points, output = sample_sin_task(N, task)

	phi = theta
	for k in range(K):
		phi = phi - alpha * #FILL IN take gradient with respect to model parameters
		#in particular, take gradient of MSE of model prediction

	test_input, test_output = sample_sin_task(M, task)
	prediction = #FILL IN prediction of model with phi on input

	return np.linalg.norm(test_output - prediction)

def MAML_HB():
	theta = #FILL IN initialize theta randomly
	updated_theta = theta
	while True: #not converged
		amplitudes = np.random.uniform(0.1, 5.0, size = J)
		phases = np.random.uniform(0, np.pi, size = J)
		for j in range(J):
			task = {'amplitude': amplitudes[j], 'phase': phases[j]}
			gradient = #FILL IN gradient of ML_point with respect to theta
			updated_theta -= beta*gradient
		theta = updated_theta