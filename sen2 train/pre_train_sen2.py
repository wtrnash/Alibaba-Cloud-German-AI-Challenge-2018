import h5py
import numpy as np
import os
import tensorflow as tf
import model
import resnet_v1
import tensorflow.contrib.slim as slim
import model_low
import vgg
import inception_resnet_v2
import pre_trained_resnet

# 标准化数据
def normalize(data):
    [one, _, _, four] = data.shape
    for i in range(one):
        for j in range(four):
            mean = np.mean(data[i, :, :, j])
            std = np.std(data[i, :, :, j])
            data[i, :, :, j] = (data[i, :, :, j] - mean) / std

    return data


# 转换成one hot
def transform_one_hot(labels):
    n_labels = np.max(labels) + 1
    one_hot = np.eye(n_labels)[labels]
    return one_hot


# 读取文件
base_dir = os.path.expanduser("./data")
path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')
path_test = os.path.join(base_dir, "round1_test_a_20181109.h5")

fid_training = h5py.File(path_training, 'r')
fid_validation = h5py.File(path_validation, 'r')
fid_test = h5py.File(path_test, 'r')

# get s1 image channel data
# it is not really loaded into memory. only the indexes have been loaded.
# 数据格式显示
print("-" * 60)
print("training part")
s1_training = fid_training['sen1']
print(s1_training.shape)
s2_training = fid_training['sen2']
print(s2_training.shape)
label_training = fid_training['label']
print(label_training.shape)

print("-" * 60)
print("validation part")
s1_validation = fid_validation['sen1']
print(s1_validation.shape)
s2_validation = fid_validation['sen2']
print(s2_validation.shape)
label_validation = fid_validation['label']
print(label_validation.shape)
val_label = np.argmax(label_validation, axis=1)

print("-" * 60)
print("test part")
s1_test = fid_test['sen1']
print(s1_test.shape)
s2_test = fid_test['sen2']
print(s2_test.shape)
print("-" * 60)

train_s1 = s1_training
train_s2 = s2_training
train_label = np.argmax(label_training, axis=1)

s1_validation = s1_validation
s2_validation = s2_validation

batch_size = 128
n_sampels = train_s1.shape[0]

# tensorflow 深度学习模型设置
class_num = 17
weight_decay = 0.01
dropout_rate = 0.5
x = tf.placeholder(tf.float32, [None, 32, 32, 10], name="x")
y_ = tf.placeholder(tf.int64, [None], name="y_")

with slim.arg_scope(pre_trained_resnet.resnet_arg_scope(weight_decay)):
    net, _ = pre_trained_resnet.resnet_v1_50(x)
    net = tf.squeeze(net, axis=[1, 2])  # 去除第一、第二个维度
    logits = slim.fully_connected(net, num_outputs=class_num,
                                  activation_fn=None, scope='predict')
# with slim.arg_scope(vgg.vgg_arg_scope(weight_decay)):
#    logits, _ = vgg.vgg_16(x, class_num, True, dropout_rate)
# logits = model.model(x, True, dropout_rate, class_num, weight_decay)
# net, _ = resnet_v1.resnet_v1_26(x)
# net = tf.squeeze(net, axis=[1, 2])  # 去除第一、第二个维度
# logits = slim.fully_connected(net, num_outputs=class_num,
#                              activation_fn=None, scope='predict')
# logits, _ = inception_resnet_v2.inception_resnet_v2(x, class_num)

variables_to_restore = slim.get_variables_to_restore()
gamma = 2
# 计算交叉熵及其平均值
with tf.name_scope('training'):
    labels = tf.one_hot(y_, class_num)
    logits = tf.nn.softmax(logits)
    logits = tf.clip_by_value(logits, 1e-10, 1.0)
    # loss = -tf.reduce_mean(tf.reduce_sum(labels * tf.log(logits)))
    loss = -tf.reduce_mean(tf.reduce_sum(labels * (1 - logits) ** gamma * tf.log(logits)))  # focal loss
    # 优化损失函数
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
    tf.summary.scalar('loss', loss)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(y_, tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# 初始化回话并开始训练过程。
with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=30)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./train_model_demo', sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)

    # 运行该函数
    init_fn = slim.assign_from_checkpoint_fn('./pre_trained_model/resnet_v1_50.ckpt', variables_to_restore,
                                             ignore_missing_vars=True)
    init_fn(sess)
    # 运行该函数

    for epochs in range(30):
        for i in range(0, n_sampels, batch_size):
            start_pos = i
            end_pos = min(i + batch_size, n_sampels)
            # train_s1_batch = np.asarray(train_s1[start_pos:end_pos, :, :, :])
            train_s2_batch = np.asarray(train_s2[start_pos:end_pos, :, :, :])
            # train_X_batch = np.concatenate((train_s1_batch, train_s2_batch), axis=3)
            train_X_batch = train_s2_batch
            train_X_batch = normalize(train_X_batch)
            label_batch = train_label[start_pos:end_pos]

            sess.run(train_step, feed_dict={x: train_X_batch, y_: label_batch})
            if (i / batch_size) % 10 == 0:
                summary, l, acc = sess.run([merged, loss, accuracy], feed_dict={x: train_X_batch, y_: label_batch})
                print("After %d training step(s), loss is %g ,accuracy is %g" % (i / batch_size, l, acc))
            train_writer.add_summary(summary, i)

        saver.save(sess, './checkpoint/pre_train_sen2_resnet50_' + str(epochs) + '.ckpt')
        # 计算验证集准确率
        pred_y = []
        train_val_y = np.argmax(label_validation, axis=1)
        batch_size = 128
        n_val_samples = s2_validation.shape[0]
        total = 0
        for i in range(0, n_val_samples, batch_size):
            start_pos = i
            end_pos = min(i + batch_size, n_val_samples)
            #  val_s1_batch = np.asarray(s1_validation[start_pos:end_pos, :, :, :])
            val_s2_batch = np.asarray(s2_validation[start_pos:end_pos, :, :, :])
            # val_X_batch = np.concatenate((val_s1_batch, val_s2_batch), axis=3)
            val_X_batch = val_s2_batch
            val_X_batch = normalize(val_X_batch)
            val_label_batch = val_label[start_pos:end_pos]

            temp = sess.run(accuracy, feed_dict={x: val_X_batch, y_: val_label_batch})
            total = total + temp * (end_pos - start_pos)

        total = total / n_val_samples
        print("第" + str(epochs) + "个epochs验证集准确率为： " + str(total))




