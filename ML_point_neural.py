from autograd import numpy as np
import matplotlib.pyplot as plt
from autograd import grad
import copy

alpha = 1e-3 #task learning rate
beta = 0.001 #meta learning rate
K = 1 #number of gradient updates in task training
N = 10*2 #number of samples used for task training
M = 10*2 #number of samples used for task testing
J = 25 #number of different tasks to train on in each iteration
meta_training_iters = 70000
deltas = []
mses = []

class MAML_HB():
	def __init__(self):
		w0 = np.random.normal(0, 0.01, size = 40) #40 by 1
		b0 = np.random.normal(0, 0.01, size = 40)
		w1 = np.random.normal(0, 0.01, size = (40,40))
		b1 = np.random.normal(0, 0.01, size = 40) 
		w2 = np.random.normal(0, 0.01, size = (1, 40))
		b2 = np.random.normal(0, 0.01, size = 1)
		self.theta = [(w0,b0), (w1,b1), (w2,b2)] #initialize theta to be random weights

	def sample_sin_task(self, N, amp, phase):
		input_points = np.random.uniform(-5., 5., size = N)
		output = amp * np.sin(input_points + phase)
		return input_points, output

	def predict(self, )
	def ML_point(self, amp, phase): #given a task, return loss after updating theta to new iteration
		input_points, output = self.sample_sin_task(N, amp, phase)

		def model_wrapper(phi):
			return model_neural_mse(phi, input_points, output)

		model_grad = grad(model_wrapper)

		phi = theta
		start_mse = model_neural_mse(phi, input_points, output)
		
		#print(sanitize(start_mse), end = ",")
		for k in range(K):
			model_grad_eval = model_grad(phi)
			phi = grad_update(phi, model_grad_eval, alpha)
			#print(sanitize(model_neural_mse(phi, input_points, output)), end = ",")
		#print()

		if train:
			return phi

		test_input, test_output = sample_sin_task(M, amp, phase)
		return model_neural_mse(phi, test_input, test_output)

	def metatrain(self): #meta-trains for 1 iteration
		#draws J tasks to train on
		amplitudes = np.random.uniform(0.1, 5.0, size = J)
		phases = np.random.uniform(0, np.pi, size = J)
		for j in range(J): #for each task, computes loss
			def ML_point_wrapper(theta):
				return ML_point(theta, amplitudes[j], phases[j])
			ML_point_grad = grad(ML_point_wrapper)
			updated_theta = grad_update(updated_theta, ML_point_grad(theta), beta)
		theta = updated_theta
		visualize(theta, iter)





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

def sanitize(autograd_obj):
	if type(autograd_obj) != np.ndarray:
		return autograd_obj._value[0]
	return autograd_obj

def ML_point(theta, amp, phase, train = False): 
	#theta is weights of network
	input_points, output = sample_sin_task(N, amp, phase)

	def model_wrapper(phi):
		return model_neural_mse(phi, input_points, output)

	model_grad = grad(model_wrapper)

	phi = theta
	start_mse = model_neural_mse(phi, input_points, output)
	
	#print(sanitize(start_mse), end = ",")
	for k in range(K):
		model_grad_eval = model_grad(phi)
		phi = grad_update(phi, model_grad_eval, alpha)
		#print(sanitize(model_neural_mse(phi, input_points, output)), end = ",")
	#print()

	if train:
		return phi

	test_input, test_output = sample_sin_task(M, task)
	return model_neural_mse(phi, test_input, test_output)

def visualize(theta, iter): #see how an inidivdual theta is doing
	task = {'amplitude': 2.5, 'phase': np.pi/2}
	train_i, train_o = sample_sin_task(N, task)
	
	phi = ML_point(theta, task, train = True)
	test_i, test_o = sample_sin_task(N, task = {'amplitude': 2.5, 'phase': np.pi/2})
	
	theta_mse = model_neural_mse(theta, test_i, test_o)
	phi_mse = model_neural_mse(phi, test_i, test_o)
	delta = theta_mse - phi_mse
	print("Iter %d, %f, %f, %f"%(iter, theta_mse, phi_mse, delta))
	deltas.append(delta)
	mses.append((theta_mse, phi_mse))
	if iter % 100 == 50:
		fig = plt.figure()
		plt.plot(deltas)
		fig.savefig("deltas_iter_%d"%iter)
		
		fig = plt.figure()
		plt.plot([x[0] for x in mses])
		plt.plot([x[1] for x in mses])
		fig.savefig("mses_iter_%d"%iter)


	# if iter > 100 and iter%5 == 0:
	# 	pred_os = []
	# 	for p in i:
	# 		pred_os.append(model_neural_pred(theta, p))
	# 	plt.scatter(pred_os, o)
	# 	plt.show()

def MAML_HB():

 
	updated_theta = copy.deepcopy(theta)
	for iter in range(meta_training_iters):
		amplitudes = np.random.uniform(0.1, 5.0, size = J)
		phases = np.random.uniform(0, np.pi, size = J)
		for j in range(J):
			def ML_point_wrapper(theta):
				return ML_point(theta, amplitudes[j], phases[j])
			ML_point_grad = grad(ML_point_wrapper)
			updated_theta = grad_update(updated_theta, ML_point_grad(theta), beta)
		theta = updated_theta
		visualize(theta, iter)

#MAML_HB()

if __name__ == "__main__":
	inp, out = sample_sin_task(100, 2., np.pi/2)
	plt.scatter(inp, out)
	plt.show()