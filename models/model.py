"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
import random, math
import tensorflow as tf

from baseline_constants import OptimLoggingKeys, AGGR_MEAN, AGGR_GEO_MED, AGGR_GAUSSIAN

from utils.model_utils import batch_data
from utils.tf_utils import graph_size

from sklearn.decomposition import PCA


class Model(ABC):

    def __init__(self, lr, seed, max_batch_size):
        self.lr = lr
        self._optimizer = None
        self.rng = random.Random(seed)

        self.graph = tf.Graph()
        if seed is not None:
            self.graph.seed = seed
        with self.graph.as_default():
            self.learning_rate_tensor = tf.compat.v1.placeholder(tf.float32, shape=[])
            self.features, self.labels, self.loss_op, self.train_op, self.eval_metric_ops = self.create_model()
            self.saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session(graph=self.graph)

        self.size = graph_size(self.graph)

        # largest batch size for which GPU will not run out of memory
        self.max_batch_size = max_batch_size if max_batch_size is not None else 2 ** 14

        with self.graph.as_default():
            self.sess.run(tf.compat.v1.global_variables_initializer())

            # metadata = tf.RunMetadata()
            # opts = tf.profiler.ProfileOptionBuilder(
            #     tf.profiler.ProfileOptionBuilder.float_operation()
            # ).with_empty_output().build()
            # self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops
        self.flops = 0

    @property
    def optimizer(self):
        """Optimizer to be used by the model."""
        if self._optimizer is None:
            self._optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate_tensor)

        return self._optimizer

    @abstractmethod
    def create_model(self):
        """Creates the model for the task.

        Returns:
            A 5-tuple consisting of:
                features: A placeholder for the samples' features.
                labels: A placeholder for the samples' labels.
                loss_op: A Tensorflow operation that, when run with the features and
                    the labels, computes the loss function on these features and models.
                train_op: A Tensorflow operation that, when run with the features and
                    the labels, trains the model.
                eval_metric_ops: A Tensorflow operation that, when run with features and labels,
                    returns the accuracy of the model.
        """
        return None, None, None, None, None

    def train(self, data, num_epochs=1, batch_size=10, lr=None):
        """
        Trains the client model.

        Args:
            data: Dict of the form {'x': [list], 'y': [list]}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training given data
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
            averaged_loss: average of stochastic loss in the final epoch
        """
        if lr is None:
            lr = self.lr
        averaged_loss = 0.0
        with self.graph.as_default():
            init_values = [self.sess.run(v) for v in tf.compat.v1.trainable_variables()]

        batched_x, batched_y = batch_data(data, batch_size, rng=self.rng, shuffle=True)
        for epoch in range(num_epochs):
            total_loss = 0.0
            for i, raw_x_batch in enumerate(batched_x):
                input_data = self.process_x(raw_x_batch)
                raw_y_batch = batched_y[i]
                target_data = self.process_y(raw_y_batch)
                with self.graph.as_default():
                    loss, _ = self.sess.run(
                        [self.loss_op, self.train_op],
                        feed_dict={self.features: input_data, self.labels: target_data, self.learning_rate_tensor: lr}
                    )
                total_loss += loss
            averaged_loss = total_loss / len(batched_x)
        with self.graph.as_default():
            update = [self.sess.run(v) for v in tf.compat.v1.trainable_variables()]
            update = [np.subtract(update[i], init_values[i]) for i in range(len(update))]
        comp = num_epochs * len(batched_y) * batch_size * self.flops
        return comp, update, averaged_loss

    def test(self, eval_data, train_data=None):
        """
        Tests the current model on the given data.

        Args:
            eval_data: dict of the form {'x': [list], 'y': [list]}
            train_data: None or same format as eval_data. If None, do not measure statistics on train_data.
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        data_lst = [eval_data] if train_data is None else [eval_data, train_data]
        output = {'eval': [-float('inf'), -float('inf')], 'train': [-float('inf'), -float('inf')]}
        for data, data_type in zip(data_lst, ['eval', 'train']):
            total_loss, total_correct, count = 0.0, 0, 0
            batched_x, batched_y = batch_data(data, self.max_batch_size, shuffle=False, eval_mode=True)
            for x, y in zip(batched_x, batched_y):
                x_vecs = self.process_x(x)
                labels = self.process_y(y)
                with self.graph.as_default():
                    loss, correct = self.sess.run(
                        [self.loss_op, self.eval_metric_ops],
                        feed_dict={self.features: x_vecs, self.labels: labels}
                    )
                total_loss += loss * len(y)  # loss returns average over batch
                total_correct += correct  # eval_op returns sum over batch
                count += len(y)
            loss = total_loss / count
            acc = total_correct / count
            output[data_type] = [loss, acc]

        return {OptimLoggingKeys.TRAIN_LOSS_KEY: output['train'][0],
                OptimLoggingKeys.TRAIN_ACCURACY_KEY: output['train'][1],
                OptimLoggingKeys.EVAL_LOSS_KEY: output['eval'][0],
                OptimLoggingKeys.EVAL_ACCURACY_KEY: output['eval'][1]
                }

    def close(self):
        self.sess.close()

    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        return np.asarray(raw_x_batch)

    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        return np.asarray(raw_y_batch)


class ServerModel:
    def __init__(self, model):
        self.model = model
        self.rng = model.rng

    @property
    def size(self):
        return self.model.size

    @property
    def cur_model(self):
        return self.model

    def send_to(self, clients):
        """Copies server model variables to each of the given clients

        Args:
            clients: list of Client objects
        """
        var_vals = {}
        with self.model.graph.as_default():
            all_vars = tf.compat.v1.trainable_variables()
            for v in all_vars:
                val = self.model.sess.run(v)
                var_vals[v.name] = val
        for c in clients:
            with c.model.graph.as_default():
                all_vars = tf.compat.v1.trainable_variables()
                for v in all_vars:
                    v.load(var_vals[v.name], c.model.sess)

    @staticmethod
    def weighted_average_oracle(points, weights):
        """Computes weighted average of atoms with specified weights

        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        tot_weights = np.sum(weights)
        weighted_updates = [np.zeros_like(v) for v in points[0]]

        for w, p in zip(weights, points):
            for j, weighted_val in enumerate(weighted_updates):
                weighted_val += (w / tot_weights) * p[j]
        return weighted_updates

    def update(self, updates, aggregation=AGGR_MEAN, max_update_norm=None, maxiter=4, visualize=False):
        """Updates server model using given client updates.

        Args:
            updates: list of (num_samples, update), where num_samples is the
                number of training samples corresponding to the update, and update
                is a list of variable weights
            aggregation: Algorithm used for aggregation. Allowed values are:
                [ 'mean', 'geom_median']
            max_update_norm: Reject updates larger than this norm,
            maxiter: maximum number of calls to the Weiszfeld algorithm if using the geometric median
        """
        def accept_update(u):
            norm = np.linalg.norm([np.linalg.norm(x) for x in u[1]])
            return not (np.isinf(norm) or np.isnan(norm))
        all_updates = updates
        updates = [u for u in updates if accept_update(u)]
        if len(updates) < len(all_updates):
            print('Rejected {} individual updates because of NaN or Inf'.format(len(all_updates) - len(updates)))
        if len(updates) == 0:
            print('All individual updates rejected. Continuing without update')
            return 1, False
        
        points = [u[1] for u in updates]
        alphas = [u[0] for u in updates]
        # print("points: shape {} type {}".format(len(points), type(points)))
        # print("points[0]: shape {} type {}".format(len(points[0]), type(points[0])))
        # print("points[0][0]: shape {} type {}".format(points[0][0].shape, type(points[0][0])))
        # print("points[0][1]: shape {} type {}".format(points[0][1].shape, type(points[0][1])))
        if visualize==True:
            self.geometric_median_update(points, alphas, maxiter=1, visualize=visualize)
            self.gaussian_aggregate(self, points, alphas, visualize=visualize)
            # federated_average_mean = self.weighted_average_oracle(points, alphas)
            print("Avg Results: {}".format(np.array(alphas)/np.sum(alphas).tolist()))
            return

        if aggregation == AGGR_MEAN:
            weighted_updates = self.weighted_average_oracle(points, alphas)
            num_comm_rounds = 1
        elif aggregation == AGGR_GEO_MED:
            weighted_updates, num_comm_rounds, _ = self.geometric_median_update(points, alphas, maxiter=maxiter, visualize=visualize)
        elif aggregation == AGGR_GAUSSIAN:
            weighted_updates = self.gaussian_aggregate(self, points, alphas, visualize=visualize)
            num_comm_rounds = 1
        else:
            raise ValueError('Unknown aggregation strategy: {}'.format(aggregation))

        update_norm = np.linalg.norm([np.linalg.norm(v) for v in weighted_updates])

        if max_update_norm is None or update_norm < max_update_norm:
            with self.model.graph.as_default():
                all_vars = tf.compat.v1.trainable_variables()
                for i, v in enumerate(all_vars):
                    init_val = self.model.sess.run(v)
                    v.load(np.add(init_val, weighted_updates[i]), self.model.sess)
            updated = True
        else:
            print('\t\t\tUpdate norm = {} is too large. Update rejected'.format(update_norm))
            updated = False

        return num_comm_rounds, updated

    def save(self, path=None):
        return self.model.saver.save(self.model.sess, path) if path is not None else None

    def load(self, path):
        return self.model.saver.restore(self.model.sess, path)

    def close(self):
        self.model.close()

    @staticmethod
    def geometric_median_update(points, alphas, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6, visualize=False):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        if visualize==True:
            # print("PCA start")
            estimator = PCA(n_components=2)
            converted = []
            for pt in points:
                point_per_client = []
                for p in pt:
                    point_per_client.append(np.array(p).flatten().tolist())
                converted.append(np.array([y for y in p for p in point_per_client]))
            # p = [pt.flatten().tolist() for pt in points]
            # print("converted: {}".format(converted))
            assert len(converted) == len(points)
            # print("converted[0].shape: {}".format(converted[0].shape))
            # print("converted[1].shape: {}".format(converted[1].shape))
            assert converted[0].shape == converted[1].shape
            pca_x = estimator.fit_transform(converted)
            # print("PCA end")
            # print("geometric median after pca: {}".format(pca_x))

        alphas = np.asarray(alphas, dtype=points[0][0].dtype) / sum(alphas)
        median = ServerModel.weighted_average_oracle(points, alphas)
        num_oracle_calls = 1

        # logging
        obj_val = ServerModel.geometric_median_objective(median, points, alphas)
        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append(log_entry)
        if verbose:
            print('Starting Weiszfeld algorithm')
            print(log_entry)

        # start
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = np.asarray([alpha / max(eps, ServerModel.l2dist(median, p)) for alpha, p in zip(alphas, points)],
                                 dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = ServerModel.weighted_average_oracle(points, weights)
            num_oracle_calls += 1
            obj_val = ServerModel.geometric_median_objective(median, points, alphas)
            log_entry = [i+1, obj_val,
                         (prev_obj_val - obj_val)/obj_val,
                         ServerModel.l2dist(median, prev_median)]
            logs.append(log_entry)
            if verbose:
                print(log_entry)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        if visualize==True:
            print("geo median weights: {}".format(weights))
        return median, num_oracle_calls, logs

    @staticmethod
    def l2dist(p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        return np.linalg.norm([np.linalg.norm(x1 - x2) for x1, x2 in zip(p1, p2)])

    @staticmethod
    def geometric_median_objective(median, points, alphas):
        """Compute geometric median objective."""
        return sum([alpha * ServerModel.l2dist(median, p) for alpha, p in zip(alphas, points)])
    
    @staticmethod
    def gaussian_aggregate(self, points, weights, visualize=False):
        """Computes gaussian aggregates 

        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        client_num = len(points)
        layer_num = len(points[0])
        #  print("points: {}".format(points))

        if visualize==True:
            print("PCA start")
            estimator = PCA(n_components=2)
            converted = []
            for pt in points:
                point_per_client = []
                for p in pt:
                    point_per_client.append(np.array(p).flatten().tolist())
                converted.append(np.array([y for y in p for p in point_per_client]))
            # p = [pt.flatten().tolist() for pt in points]
            # print("converted: {}".format(converted))
            assert len(converted) == len(points)
            print("converted[0].shape: {}".format(converted[0].shape))
            print("converted[1].shape: {}".format(converted[1].shape))
            assert converted[0].shape == converted[1].shape
            pca_x = estimator.fit_transform(converted)
            print("PCA end")
            print("gaussian aggregation after pca: {}".format(pca_x))



        server_weights = []
        with self.model.graph.as_default():
            all_vars = tf.compat.v1.trainable_variables()
            for v in all_vars:
                val = self.model.sess.run(v)
                server_weights.append(val)

        distances = []
        gaussian_weight = []
        miu = []
        sigma = []
        def norm(arr):
            a = np.exp(arr)
            return a/np.sum(a)
        # client_weights = []
        for i_layer in range(layer_num):
            thislayer = []
            for i_client in range(client_num):
                thislayer.append(points[i_client][i_layer])
            # client_weights.append(thislayer)
            
            server_this_layer = server_weights[i_layer]
            assert(server_this_layer.shape==thislayer[0].shape)
            assert(len(thislayer)==client_num)
            # distances.append([ServerModel.l2dist(l.flatten(), server_this_layer.flatten()) for l in thislayer])
            distances.append([(np.linalg.norm(np.ravel(l) - np.ravel(server_this_layer), ord=2)) for l in thislayer])
            # if visualize:
            # print("distance : {}".format(distances[i_layer]))
            assert(len(distances[i_layer])==client_num)
            miu.append((np.max(distances[i_layer])+np.min(distances[i_layer]))/2)
            s = 0
            for d in distances[i_layer]:
                s += pow(d-miu[i_layer], 2)
            s = pow(s, 0.5)
            sigma.append(s)

            gaussian_weight_thislayer = []
            for i_client in range(client_num):
                gaussian_weight_thislayer.append(
                    math.exp(-pow((distances[i_layer][i_client]-miu[i_layer]),2)/(2*pow(sigma[i_layer], 2)))/(s*pow(2*math.pi, 0.5))
                )
            # if visualize:
            normed_weight = norm(gaussian_weight_thislayer)
            # print("gaussian before: {}".format(gaussian_weight_thislayer))
            # print("gaussian normed: {}".format(normed_weight.tolist()))
            gaussian_weight.append(normed_weight)

        # global_distance = [ServerModel.l2dist(np.array(server_weights).flatten(), np.array(pt).flatten()) for pt in points]
        # print("global distance: {}".format(global_distance))
        weighted_updates = [np.zeros_like(v) for v in points[0]]
        for i_layer in range(layer_num):
            # 3 parameters: 
            # client[i]'s parameter: points[i][i_layer]
            # gaussain parameter: gaussian[i_layer]
            # sample rate parsameter: weights[i]/tot_weights
            thislayer_gaussian_weight = gaussian_weight[i_layer]
            for i_client in range(client_num):
                weighted_updates[i_layer] += points[i_client][i_layer]*thislayer_gaussian_weight[i_client]
        return weighted_updates
