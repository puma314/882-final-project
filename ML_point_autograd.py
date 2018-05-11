from autograd import numpy as np
import matplotlib.pyplot as plt
from autograd import grad

alpha = 0.01 #task learning rate
beta = 0.001 #meta learning rate
K = 1 #number of gradient updates in task training
N = 5 #number of samples used for task training
M = 10 #number of samples used for task testing
J = 10 #number of different tasks to train on in each iteration

def sample_sin_task(N, task):
	input_points = np.random.uniform(-10., 10., size = N)
	A = task['amplitude']
	P = task['phase']
	output = A * np.sin(input_points + P)
	return input_points, output

def model(theta, inp, output): #assume theta is a matrix, let it be a 2 variable LR
	mat = np.vstack((inp, np.ones(len(inp))))
	return np.linalg.norm(np.dot(theta, mat)-output)

def ML_point(theta, task): 
	#theta is weights of network
	input_points, output = sample_sin_task(N, task)
	def model_wrapper(phi):
		return model(phi, input_points, output)

	phi = theta
	for k in range(K):
		model_grad = grad(model_wrapper)
		phi = phi - alpha * model_grad(phi)

	test_input, test_output = sample_sin_task(M, task)
	return model(phi, test_input, test_output)

def MAML_HB():
	theta = np.array([0., 0.])
	updated_theta = theta
	while True: #not converged
		amplitudes = np.random.uniform(0.1, 5.0, size = J)
		phases = np.random.uniform(0, np.pi, size = J)
		for j in range(J):
			task = {'amplitude': amplitudes[j], 'phase': phases[j]}
			def ML_point_wrapper(theta):
				return ML_point(theta, task)
			ML_point_grad = grad(ML_point_wrapper)
			updated_theta -= beta*ML_point_grad(theta)
		theta = updated_theta
		input(theta)

MAML_HB()