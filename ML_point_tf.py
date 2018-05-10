#from livelossplot import PlotLosses
import tensorflow as tf
import numpy as np

# MAML parameters
alpha = 1e-3        # task learning rate
beta = 0.001        # meta learning rate
K = 1               # number of gradient updates in task training
N = 10              # number of samples used for task training
M = 10              # number of samples used for task testing
J = 25              # number of different tasks to train on in each iteration
meta_training_iters = 50000

# Network parameters
n_fc = 40

class MAML_HB():
    def __init__(self):
        self.theta = {
            "w1": tf.Variable(tf.truncated_normal([1, n_fc], stddev=0.1), name="w1"),
            "b1": tf.Variable(tf.constant(0.1, shape=[n_fc]), name="b1"),
            "w2": tf.Variable(tf.truncated_normal([n_fc, n_fc], stddev=0.1), name="w2"),
            "b2": tf.Variable(tf.constant(0.1, shape=[n_fc]), name="b2"),
            "out": tf.Variable(tf.truncated_normal([n_fc, 1], stddev=0.01), name="out"),
            "outb": tf.Variable(tf.truncated_normal([1], stddev=0.01), name="outb")

        }

        self.tasks = [tf.placeholder(tf.float32, shape=(2,), name="input_task") for _ in range(J)]
        self.train_op, self.loss = self.build_train_op()
    
    def ML_point(self, task):
        with tf.name_scope("ML_point"):
            amplitude, phase = tf.unstack(task)
            input_pts, output_pts = sample_sin_task_pts(N, amplitude, phase)

            phi = {}

            with tf.name_scope("train"):
                # Initialize phi with the first gradient update
                pred = self.forward_pass(input_pts, self.theta)
                loss = mse(pred, output_pts)
                loss = tf.Print(loss, [loss])
                grad = tf.gradients(loss, list(self.theta.values()))
                grad = dict(zip(self.theta.keys(), grad))
                phi = dict(zip(self.theta.keys(), [self.theta[key] - alpha * grad[key] for key in self.theta.keys()]))

                for k in range(K-1):
                    pred = self.forward_pass(input_pts, phi)
                    loss = mse(pred, output_pts)

                    grad = tf.gradients(loss, list(phi.values()))
                    grad = dict(zip(phi.keys(), grad))

                    phi = dict(zip(phi.keys(), [phi[key] - alpha * grad[key] for key in phi.keys()])) 
 
            with tf.name_scope("test"):
                test_input_pts, test_output_pts = sample_sin_task_pts(M, amplitude, phase)
                test_pred = self.forward_pass(test_input_pts, phi)
                return mse(test_pred, test_output_pts)

    def forward_pass(self, inp, params):
        with tf.name_scope("model"):
            fc1 = tf.add(tf.matmul(inp, params["w1"]), params["b1"])
            fc1 = tf.nn.relu(fc1)
            fc2 = tf.add(tf.matmul(fc1, params["w2"]), params["b2"])
            fc2 = tf.nn.relu(fc2)
            out = tf.add(tf.matmul(fc2, params["out"]), params["outb"])

            self._summarize_variables()

            return out

    def build_train_op(self):
        " One iter of the outer loop. "
        with tf.name_scope("outer_loop"):
            task_losses = []
            for i, task in enumerate(self.tasks):
                task_loss = self.ML_point(task)
                task_losses.append(task_loss)
            loss = tf.add_n(task_losses) / tf.to_float(J)

            optimizer = tf.train.AdamOptimizer(learning_rate=beta)
            train_op = optimizer.minimize(loss, var_list=list(self.theta.values()))
            
            return train_op, loss

    def _summarize_variables(self):
        with tf.name_scope("summaries"):
            with tf.name_scope("w"):
                tf.summary.scalar("mean", tf.reduce_mean(self.theta["w1"]))
                tf.summary.histogram("histogram", self.theta["w1"])
                tf.summary.scalar("mean", tf.reduce_mean(self.theta["w2"]))
                tf.summary.histogram("histogram", self.theta["w2"])
            with tf.name_scope("b"):
                tf.summary.scalar("mean", tf.reduce_mean(self.theta["b1"]))
                tf.summary.histogram("histogram", self.theta["b1"])
                tf.summary.scalar("mean", tf.reduce_mean(self.theta["b2"]))
                tf.summary.histogram("histogram", self.theta["b2"])

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
    return tf.transpose(input_points), tf.transpose(output_points)

def mse(pred, actual):
    return tf.reduce_mean(tf.squared_difference(pred, actual)) 
   
def main():
    sess = tf.InteractiveSession()
    maml = MAML_HB()
    merged_summary = tf.summary.merge_all()

    tf.global_variables_initializer().run()

    train_writer = tf.summary.FileWriter("logs", sess.graph)

    for i in range(meta_training_iters):
        tasks = draw_sin_tasks(J)
        summary, _, loss = sess.run([merged_summary, maml.train_op, maml.loss], feed_dict={tp: task for tp, task in zip(maml.tasks, tasks)})
        theta = maml.theta
        train_writer.add_summary(summary, i)
        if i % 100 == 0:
            print("Iter {}:".format(i), loss)
            print("Theta: ", theta)
            #print("bef: ", bef[0, :5])
            #print("aft: ", aft[0, :5])
            #print("aft_phi: ", aft_phi[0, :5])
            
    graph = tf.get_default_graph()
    writer = tf.summary.FileWriter("logs")
    writer.add_graph(graph=graph)


if __name__ == "__main__":
    main()
