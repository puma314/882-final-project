import tensorflow as tf
import numpy as np

# MAML parameters
alpha = 1e-3        # task learning rate
beta = 0.001        # meta learning rate
K = 1               # number of gradient updates in task training
N = 10*2            # number of samples used for task training
M = 10*2            # number of samples used for task testing
J = 25              # number of different tasks to train on in each iteration
meta_training_iters = 70000

# Network parameters
n_fc = 40

class MAML_HB():
    def __init__(self):
        self.theta = {
            "w1": tf.Variable(tf.truncated_normal([N, n_fc], stddev=0.01)),
            "b1": tf.Variable(tf.zeros([n_fc])),
            "w2": tf.Variable(tf.truncated_normal([n_fc, n_fc], stddev=0.01)),
            "b2": tf.Variable(tf.zeros([n_fc])),
            "out": tf.Variable(tf.truncated_normal([n_fc, N], stddev=0.01))
        }

        self.tasks = [tf.placeholder(tf.float32, shape=(2,)) for _ in range(J)]
        self.train_op, self.loss = self.build_train_op()
    
    def ML_point(self, task):
        amplitude, phase = tf.unstack(task)
        input_pts, output_pts = sample_sin_task_pts(N, amplitude, phase)

        phi = {}
        for key, val in self.theta.items():
            #phi[key] = tf.Variable(tf.zeros(val.get_shape()))
            phi[key] = val

        for k in range(K):
            pred = self.forward_pass(input_pts, phi)
            loss = mse(pred, output_pts) # higher loss means lower negative logprob
            
            grad = tf.gradients(loss, list(phi.values()))
            grad = dict(zip(phi.keys(), grad))

            for key, val in phi.items():
                phi_update = tf.assign(val, val - alpha * grad[key])

            #import pdb; pdb.set_trace()
    
        test_input_pts, test_output_pts = sample_sin_task_pts(M, amplitude, phase)
        test_pred = self.forward_pass(test_input_pts, phi)
        return mse(test_pred, test_output_pts)

    def forward_pass(self, inp, params):
        fc1 = tf.add(tf.matmul(inp, params["w1"]), params["b1"])
        fc1 = tf.nn.relu(fc1)
        fc2 = tf.add(tf.matmul(fc1, params["w2"]), params["b2"])
        fc2 = tf.nn.relu(fc2)
        out = tf.matmul(fc2, params["out"]) + params["out"]
        return out

    def build_train_op(self):
        " One iter of the outer loop. "
        task_losses = []
        for task in self.tasks:
            task_loss = self.ML_point(task)
            task_losses.append(task_loss)
        loss = tf.add_n(task_losses)
        print("Loss shape:", task_losses)
        grad = tf.gradients(loss, list(self.theta.values()))
        grad = dict(zip(self.theta.keys(), grad))
        return tf.group(*[tf.assign(val, val - beta * grad[key]) for key, val in self.theta.items()]), loss


def draw_sin_tasks(J):
    " Returns a set of sampled sin tasks (amplitude, phase). "
    return [
        np.array([
            np.random.uniform(0.1, 5.0),
            np.random.uniform(0.0, np.pi)
        ]) 
        for _ in range(J)
    ]

def sample_sin_task_pts(N, amplitude, phase):
    " Given sin task params (amplitude, phase), returns N observations sampled from the task. "
    input_points = tf.random_uniform((1, N), minval=-10., maxval=10.)
    output_points = amplitude * tf.sin(input_points + phase)
    return input_points, output_points

def mse(pred, actual):
    return tf.reduce_mean(tf.squared_difference(pred, actual)) 

def eval_theta(theta):
    task = {"amplitude": 2, "phase": np.pi/2}
    train_inp, train_out = sample_sin_task(50, task)
    phi = ML_point(theta, task, train=True)
    test_inp, test_out = sample_sin_task(100, task = {"amplitude": 2, "phase": np.pi/2})
    preds = toy_model(test_inp, phi) 
    return mse(preds, test_out)
    
def main():
    sess = tf.InteractiveSession()
    maml = MAML_HB()
    tf.global_variables_initializer().run()
    for i in range(meta_training_iters):
        tasks = draw_sin_tasks(J)
        _, loss = sess.run([maml.train_op, maml.loss], feed_dict={tp: task for tp, task in zip(maml.tasks, tasks)})
        print("Loss:", loss)
            
    graph = tf.get_default_graph()
    writer = tf.summary.FileWriter("logs")
    writer.add_graph(graph=graph)


if __name__ == "__main__":
    main()
