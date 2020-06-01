import tensorflow as tf

from model import Model

IMAGE_SIZE = 28


class ClientModel(Model):
    def __init__(self, lr, num_classes, max_batch_size=None, seed=None):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(lr, seed, max_batch_size)

    def create_model(self):
        """Model function for CNN."""
        features = tf.compat.v1.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name="features")
        labels = tf.compat.v1.placeholder(tf.int64, shape=[None], name="labels")
        input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        conv1 = tf.compat.v1.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool1 = tf.compat.v1.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.compat.v1.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.compat.v1.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.compat.v1.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.relu)
        logits = tf.compat.v1.layers.dense(inputs=dense, units=self.num_classes)
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        with tf.control_dependencies([loss]):
            train_op = self.optimizer.minimize(
                loss=loss,
                global_step=tf.compat.v1.train.get_global_step())
        eval_metric_ops = tf.math.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, loss, train_op, eval_metric_ops
