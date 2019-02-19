import tensorflow as tf


def model(x, is_training, dropout_pro, num, weight_decay):
    input_layer = tf.reshape(x, [-1, 32, 32, 10])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        kernel_initializer=tf.glorot_uniform_initializer(),
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        name="2/conv1")

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        kernel_initializer=tf.glorot_uniform_initializer(),
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        name="2/conv2")

    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        kernel_initializer=tf.glorot_uniform_initializer(),
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        name="2/conv3")

    pool1 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, name="2/pool1")  # 16 * 16 * 128

    conv4 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        kernel_initializer=tf.glorot_uniform_initializer(),
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        name="2/conv4")

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        kernel_initializer=tf.glorot_uniform_initializer(),
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        name="2/conv5")

    shape = conv5.get_shape()

    flat = tf.reshape(conv5, [-1, shape[1].value * shape[2].value * shape[3].value])

    dense1 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu,
                             kernel_initializer=tf.glorot_uniform_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             name="2/dense1")

    dropout1 = tf.layers.dropout(inputs=dense1, rate=dropout_pro, training=is_training, name="2/dropout1")

    dense2 = tf.layers.dense(inputs=dropout1, units=2048, activation=tf.nn.relu,
                             kernel_initializer=tf.glorot_uniform_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                             name="2/dense2")

    dropout2 = tf.layers.dropout(inputs=dense2, rate=dropout_pro, training=is_training, name="2/dropout2")

    y = tf.layers.dense(inputs=dropout2, units=num, activation=None,
                        kernel_initializer=tf.glorot_uniform_initializer(), name="2/y")
    return y
