from autograd import numpy as np
import matplotlib.pyplot as plt
from autograd import grad
import copy

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

def model_linear(theta, inp, output): #assume theta is a matrix, let it be a 2 variable LR
	mat = np.vstack((inp, np.ones(len(inp))))
	return np.linalg.norm(np.dot(theta, mat)-output)

def model_neural_pred(theta, inputs):
	for i, (W, b) in enumerate(theta):
		#print(W.shape, b. shape, inputs.shape)
		outputs = np.dot(W, inputs) + b
		inputs = np.maximum(outputs, 0)
	return outputs

def model_neural_mse(theta, inp, output):
	mse = 0
	for inp_point, out_point in zip(inp, output):
		mse += (out_point - model_neural_pred(theta, np.array(inp_point)))**2
	return mse

def grad_update(theta, grads, alpha):
	update = []
	for i in range(len(theta)):
		inter = []
		for j in range(len(theta[i])):
			inter.append(theta[i][j] - alpha * grads[i][j])
		update.append(tuple(inter))
	return update

def ML_point(theta, task, train = False): 
	#theta is weights of network
	input_points, output = sample_sin_task(N, task)
	def model_wrapper(phi):
		return model_neural_mse(phi, input_points, output)

	phi = theta
	for k in range(K):
		model_grad = grad(model_wrapper)
		model_grad_eval = model_grad(phi)
		phi = grad_update(phi, model_grad_eval, alpha)

	if train:
		return phi

	test_input, test_output = sample_sin_task(M, task)
	return model_neural_mse(phi, test_input, test_output)

def visualize(theta, iter): #see how an inidivdual theta is doing
	task = {'amplitude': 2.5, 'phase': np.pi/2}
	train_i, train_o = sample_sin_task(50, task)
	phi = ML_point(theta, task, train = True)
	test_i, test_o = sample_sin_task(100, task = {'amplitude': 2.5, 'phase': np.pi/2})
	print(model_neural_mse(phi, test_i, test_o))
	# if iter > 100 and iter%5 == 0:
	# 	pred_os = []
	# 	for p in i:
	# 		pred_os.append(model_neural_pred(theta, p))
	# 	plt.scatter(pred_os, o)
	# 	plt.show()

def MAML_HB():
	w0 = np.random.normal(0, 0.01, size = 40) #40 by 1
	b0 = np.random.normal(0, 0.01, size = 40)
	w1 = np.random.normal(0, 0.01, size = (40,40))
	b1 = np.random.normal(0, 0.01, size = 40) 
	w2 = np.random.normal(0, 0.01, size = (1, 40))
	b2 = np.random.normal(0, 0.01, size = 1)
	theta = [(w0,b0), (w1,b1), (w2,b2)]

	updated_theta = copy.deepcopy(theta)
	for iter in range(1000):
		amplitudes = np.random.uniform(0.1, 5.0, size = J)
		phases = np.random.uniform(0, np.pi, size = J)
		for j in range(J):
			task = {'amplitude': amplitudes[j], 'phase': phases[j]}
			def ML_point_wrapper(theta):
				return ML_point(theta, task)
			ML_point_grad = grad(ML_point_wrapper)
			updated_theta = grad_update(updated_theta, ML_point_grad(theta), beta)
		theta = updated_theta
		visualize(theta, iter)

MAML_HB()