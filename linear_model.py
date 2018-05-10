#from livelossplot import PlotLosses
import tensorflow as tf
import numpy as np

# MAML parameters
alpha = 1e-3        # task learning rate
beta = 0.001        # meta learning rate
K = 1               # number of gradient updates in task training
N = 5              # number of samples used for task training
M = 5              # number of samples used for task testing
J = 10              # number of different tasks to train on in each iteration
meta_training_iters = 50000

# Network parameters
n_fc = 40
tau = 0.1

def tensors_to_column(tensors):
    if isinstance(tensors, (tuple, list)):
        return tf.concat(tuple(tf.reshape(tensor, [-1, 1]) for tensor in tensors), axis=0)
    else:
        return tf.reshape(tensors, [-1, 1])

def column_to_tensors(tensors_template, colvec):
    with tf.name_scope("column_to_tensors"):
        if isinstance(tensors_template, (tuple, list)):
            offset = 0
            tensors = []
            for tensor_template in tensors_template:
                sz = np.prod(tensor_template.shape.as_list(), dtype=np.int32)
                tensor = tf.reshape(colvec[offset:(offset + sz)], tensor_template.shape)
                tensors.append(tensor)
                offset += sz

            tensors = tuple(tensors)
        else:
            tensors = tf.reshape(colvec, tensors_template.shape)

        return tensors

class MAML_HB():
    def __init__(self):
        self.theta = {
            "w1": tf.Variable(tf.truncated_normal([n_fc,1], stddev=0.1), name="w1"),
            "b1": tf.Variable(tf.constant(0.1, shape=[1]), name="b1"),
        }

        self.tasks = [tf.placeholder(tf.float32, shape=(n_fc,1), name="input_task") for _ in range(J)]
        self.train_op, self.loss = self.build_train_op()
    
    def ML_point(self, task):
        with tf.name_scope("ML_point"):
            task_phi = task
            input_pts, output_pts = sample_linear_task_pts(N, task_phi)

            phi = {}

            with tf.name_scope("train"):
                # Initialize phi with the first gradient update
                pred = self.forward_pass(input_pts, self.theta)
                loss = mse(pred, output_pts)
                loss = tf.Print(loss, [loss])
                grad = tf.gradients(loss, list(self.theta.values()))
                #phi = dict(zip(self.theta.keys(), [self.theta[key] + 0. for key in self.theta.keys()]))
                #keys, vals = zip(*[(k, v) for k, v in phi.items()])
                #og_flat_params = tf.squeeze(tensors_to_column(vals))
                
                grad = dict(zip(self.theta.keys(), grad))
                phi = dict(zip(self.theta.keys(), [self.theta[key] - alpha * grad[key] for key in self.theta.keys()]))

                keys, vals = zip(*[(k, v) for k, v in phi.items()])
                flat_params = tf.squeeze(tensors_to_column(vals))
                phi = column_to_tensors(vals, flat_params)
                phi = {keys[i]: phi[i] for i in range(len(phi))}

            with tf.name_scope("test"):
                test_input_pts, test_output_pts = sample_linear_task_pts(M, task_phi)
                test_pred = self.forward_pass(test_input_pts, phi)
                test_mse = mse(test_pred, test_output_pts)

                log_pr_hessian = tf.hessians(test_mse, flat_params)
                log_prior_hessian = tf.eye(n_fc + 1) * tau
                hessian = tf.add(log_prior_hessian, log_pr_hessian)
                
                #test_mse = tf.Print(test_mse, [log_pr_hessian], message = "Log Pr Hessian")
                #test_mse = tf.Print(test_mse, [tf.linalg.logdet(hessian)], message = "Log det")
                
                return test_mse
                #return tf.add(test_mse, tf.linalg.logdet(hessian))


    def forward_pass(self, inp, params):
        with tf.name_scope("model"):
            fc1 = tf.add(tf.matmul(inp, params["w1"]), params["b1"])
            self._summarize_variables()
            return fc1

    def build_train_op(self):
        " One iter of the outer loop. "
        with tf.name_scope("outer_loop"):
            task_losses = []
            for i, task in enumerate(self.tasks):
                task_loss = self.ML_point(task)
                task_losses.append(task_loss)
            loss = tf.add_n(task_losses) / tf.to_float(J)

            #grad = tf.gradients(loss, list(self.theta.values()))
            #grad = dict(zip(self.theta.keys(), grad))
            #self.theta = dict(zip(self.theta.keys(), [self.theta[key] - alpha * grad[key] for key in self.theta.keys()]))

            #print(grad)
            #with tf.control_dependencies([grad]):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = beta)
            #optimizer = tf.train.AdamOptimizer(learning_rate=beta)
            train_op = optimizer.minimize(loss, var_list=list(self.theta.values()))
            
            return train_op, loss

    def _summarize_variables(self):
        with tf.name_scope("summaries"):
            with tf.name_scope("w"):
                tf.summary.scalar("mean", tf.reduce_mean(self.theta["w1"]))
                tf.summary.histogram("histogram", self.theta["w1"])
            with tf.name_scope("b"):
                tf.summary.scalar("mean", tf.reduce_mean(self.theta["b1"]))
                tf.summary.histogram("histogram", self.theta["b1"])

def draw_phi_tasks(J, theta):
    " Returns a set of sampled sin tasks (amplitude, phase). "
    return [
        np.random.multivariate_normal(mean = np.squeeze(theta),
            cov = 1./tau * np.eye(n_fc)).reshape(-1, 1)
        for _ in range(J)
    ]

def sample_linear_task_pts(N, phi):
    "Given a phi, randomly generates N inputs and creates output vector"
    input_points = tf.random_uniform((N, n_fc), minval=-10., maxval=10.)
    #randomly generated design matrix
    output_points = tf.matmul(input_points, phi)
    return input_points, output_points

def mse(pred, actual):
    return tf.reduce_mean(tf.squared_difference(pred, actual)) 

def main():
    sess = tf.InteractiveSession()
    maml = MAML_HB()
    merged_summary = tf.summary.merge_all()

    tf.global_variables_initializer().run()

    train_writer = tf.summary.FileWriter("logs", sess.graph)

    theta = tf.random_uniform((n_fc,1), minval = 5., maxval = 7.)
    theta = np.array([5., -1., 2., 0.]*10)
    assert(len(theta) == n_fc)
    for i in range(500):
        tasks = draw_phi_tasks(J, theta)
        summary, _, loss = sess.run([merged_summary, maml.train_op, maml.loss], feed_dict={tp: task for tp, task in zip(maml.tasks, tasks)})
        train_writer.add_summary(summary, i)
        if i % 1 == 0:
            print("Iter {}:".format(i), loss)

    import pdb; pdb.set_trace()
            
    graph = tf.get_default_graph()
    writer = tf.summary.FileWriter("logs")
    writer.add_graph(graph=graph)


if __name__ == "__main__":
    main()
