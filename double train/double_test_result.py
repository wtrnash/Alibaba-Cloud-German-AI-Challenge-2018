import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report
import pandas as pd
import tensorflow as tf
import model1
import model2
import model_low2
import model_low1
import resnet_v1
import tensorflow.contrib.slim as slim

# 标准化数据
def normalize(data):
    [one, _, _, four] = data.shape
    for i in range(one):
        for j in range(four):
            mean = np.mean(data[i, :, :, j])
            std = np.std(data[i, :, :, j])
            data[i, :, :, j] = (data[i, :, :, j] - mean) / std

    return data


# 读取文件
base_dir = os.path.expanduser("./data")
path_validation = os.path.join(base_dir, 'validation.h5')

fid_validation = h5py.File(path_validation, 'r')


# get s1 image channel data
# it is not really loaded into memory. only the indexes have been loaded.
# 数据格式显示

print("-" * 60)
print("validation part")
s1_validation = fid_validation['sen1']
print(s1_validation.shape)
s2_validation = fid_validation['sen2']
print(s2_validation.shape)
label_validation = fid_validation['label']
print(label_validation.shape)
val_label = np.argmax(label_validation, axis=1)

batch_size = 100
n_val_samples = s2_validation.shape[0]

# tensorflow 深度学习模型设置
class_num = 17
x1 = tf.placeholder(tf.float32, [None, 32, 32, 8], name="x1")
x2 = tf.placeholder(tf.float32, [None, 32, 32, 10], name="x2")
y_ = tf.placeholder(tf.int64, [None], name="y_")


# net, _ = resnet_v1.resnet_v1_26(x1, is_training=False)
# net = tf.squeeze(net, axis=[1, 2])  # 去除第一、第二个维度
# logits1 = slim.fully_connected(net, num_outputs=class_num,
#                               activation_fn=None, scope='predict')

logits1 = model1.model(x1, False, 0.0, class_num, 0.0)
logits2 = model2.model(x2, False, 0.0, class_num, 0.0)

# sen1 和 sen2的权重
sen1_weight = 0.1
# 计算交叉熵及其平均值
with tf.name_scope('training'):
    labels = tf.one_hot(y_, class_num)
    logits = tf.nn.softmax(logits1 + logits2)
    logits = tf.clip_by_value(logits, 1e-10, 1.0)
    loss = -tf.reduce_mean(tf.reduce_sum(labels * tf.log(logits)))
    # 优化损失函数
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
    tf.summary.scalar('loss', loss)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(y_, tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

split_index = 21000
# 初始化回话并开始训练过程。
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, './checkpoint/val_double_model_10.ckpt')

    # 计算验证集准确率
    total = 0
    for i in range(split_index, n_val_samples, batch_size):
        start_pos = i
        end_pos = min(i + batch_size, n_val_samples)
        val_s1_batch = np.asarray(s1_validation[start_pos:end_pos, :, :, :])
        val_s2_batch = np.asarray(s2_validation[start_pos:end_pos, :, :, :])
        val_s1_batch = normalize(val_s1_batch)
        val_s2_batch = normalize(val_s2_batch)
        val_label_batch = val_label[start_pos:end_pos]

        temp = sess.run(accuracy, feed_dict={x1: val_s1_batch, x2: val_s2_batch, y_: val_label_batch})
        total = total + temp * (end_pos - start_pos)


total = total / (n_val_samples - split_index)
print("验证集准确率为： " + str(total))