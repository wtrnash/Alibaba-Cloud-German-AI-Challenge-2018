import tensorflow as tf


def model(x, is_training, dropout_pro, num, weight_decay):
    with tf.name_scope('model2'):
        input_layer = tf.reshape(x, [-1, 32, 32, 10])
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            name="model2/conv1")

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            name="model2/conv2")

        conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            name="model2/conv3")

        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            name="model2/conv4")

        pool1 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, name="model2/pool1")    # 16 * 16 * 128

        conv5 = tf.layers.conv2d(
            inputs=pool1,
            filters=256,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            name="model2/conv5")

        conv6 = tf.layers.conv2d(
            inputs=conv5,
            filters=256,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            name="model2/conv6")

        conv7 = tf.layers.conv2d(
            inputs=conv6,
            filters=256,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            name="model2/conv7")

        pool2 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2, name="model2/pool3")    # 8 * 8 * 256

        conv8 = tf.layers.conv2d(
            inputs=pool2,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            name="model2/conv8")

        conv9 = tf.layers.conv2d(
            inputs=conv8,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            name="model2/conv9")

        conv10 = tf.layers.conv2d(
            inputs=conv9,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            name="model2/conv10")

        pool3 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2, name="model2/pool4")  # 4 * 4 * 512

        conv11 = tf.layers.conv2d(
            inputs=pool3,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            name="model2/conv11")

        conv12 = tf.layers.conv2d(
            inputs=conv11,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            name="model2/conv12")

        conv13 = tf.layers.conv2d(
            inputs=conv12,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=tf.glorot_uniform_initializer(),
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            name="model2/conv13")

        shape = conv13.get_shape()

        flat = tf.reshape(conv13, [-1, shape[1].value * shape[2].value * shape[3].value])

        dense1 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_uniform_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 name="model2/dense1")

        dropout1 = tf.layers.dropout(inputs=dense1, rate=dropout_pro, training=is_training, name="model2/dropout1")

        dense2 = tf.layers.dense(inputs=dropout1, units=1024, activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_uniform_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 name="model2/dense2")

        dropout2 = tf.layers.dropout(inputs=dense2, rate=dropout_pro, training=is_training, name="model2/dropout2")

        y = tf.layers.dense(inputs=dropout2, units=num, activation=None,
                            kernel_initializer=tf.glorot_uniform_initializer(), name="model2/y")
    return y
