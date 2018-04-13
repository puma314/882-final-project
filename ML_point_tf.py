import tensorflow as tf
import numpy as np

alpha = 1e-3 #task learning rate
beta = 0.001 #meta learning rate
K = 1 #number of gradient updates in task training
N = 10*2 #number of samples used for task training
M = 10*2 #number of samples used for task testing
J = 25 #number of different tasks to train on in each iteration
meta_training_iters = 70000

def sample_sin_task(N, task):
    # Sample observations from task
    input_points = tf.random_uniform((N,), minval=-10, maxval=10)
    output_points = tf.scalar_mul(task['amplitude'], tf.sin(input_points + task['phase']))
    return input_points, output_points

def mse(pred, actual):
    return tf.reduce_mean(tf.squared_difference(pred, actual)) 

def ML_point(theta, task, train=False):
    input_pts, output_pts = sample_sin_task(N, task)
    import pdb; pdb.set_trace()
    phi = {
        "w1": tf.Variable(theta["w1"].initialized_value()),
        "b1": tf.Variable(theta["b1"].initialized_value())
    }
    for k in range(K):
        pred = toy_model(input_pts, phi) 
        grad = dict(zip(phi.keys(), tf.gradients(mse(pred, output_pts), phi.values())))
        phi = dict(zip(phi.keys(), [phi[key] - alpha * grad[key] for key in phi.keys()]))
    
    if train:
        return phi

    test_input, test_output = sample_sin_task(M, task)
    test_pred = toy_model(test_input, phi)
    return mse(test_pred, test_output)  

def MAML_HB():
    theta = {
        "w1": tf.Variable(tf.random_normal([40, 1])),
        "b1": tf.Variable(tf.zeros([40]))
    }

    curr_theta = {
        "w1": tf.Variable(theta["w1"].initialized_value()),
        "b1": tf.Variable(theta["b1"].initialized_value())
    }

    for i in range(meta_training_iters):
        amplitudes = tf.random_uniform((J,), minval=0.1, maxval=5.0)
        phases = tf.random_uniform((J,), minval=0.0, maxval=np.pi)
        for j in range(J):
            task = {'amplitude': amplitudes[j], 'phase': phases[j]}
            grad = dict(zip(curr_theta.keys(), tf.gradients(ML_point(curr_theta, task), curr_theta.values())))
            # Update each param in theta
            curr_theta = dict(zip(curr_theta.keys(), [curr_theta[key] - beta * grad[key] for key in curr_theta.keys()]))
        theta = curr_theta
        print("Iter {}: mse={}".format(i, eval_theta(theta)))

def toy_model(inp, weights):
    print("inp:", inp)
    print(inp.shape)
    return tf.matmul(weights["w1"], inp) + weights["b1"]

def eval_theta(theta):
    task = {"amplitude": 2, "phase": np.pi/2}
    train_inp, train_out = sample_sin_task(50, task)
    phi = ML_point(theta, task, train=True)
    test_inp, test_out = sample_sin_task(100, task = {"amplitude": 2, "phase": np.pi/2})
    preds = toy_model(test_inp, phi) 
    return mse(preds, test_out)
    
def main():
    sess = tf.InteractiveSession()
    
    tf.global_variables_initializer().run()

    maml = MAML_HB()   

if __name__ == "__main__":
    main()
