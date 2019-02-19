import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report
import pandas as pd
import tensorflow as tf
import model
import resnet_v1
import tensorflow.contrib.slim as slim
import model_low
import inception_resnet_v2


# 上下翻转
def flip_up_to_bottom(picture):
    result = np.array(picture)
    [rows, _, _] = result.shape
    for i in range(int(rows / 2)):
        result[[i, rows - i - 1], :, :] = result[[rows - i - 1, i], :, :]   # 用python的交换会失败

    return result


# 左右翻转
def flip_right_to_left(picture):
    result = np.array(picture)
    [_, columns, _] = result.shape
    for i in range(int(columns / 2)):
        result[:, [i, columns - i - 1], :] = result[:, [columns - i - 1, i], :]  # 用python的交换会失败

    return result


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

batch_size = 50
n_val_samples = s2_validation.shape[0]

# tensorflow 深度学习模型设置
class_num = 17
x = tf.placeholder(tf.float32, [None, 32, 32, 10], name="x")
y_ = tf.placeholder(tf.int64, [None], name="y_")
# logits = model.model(x, False, 0.0, class_num, 0.0)
# net, _ = resnet_v1.resnet_v1_50(x, is_training=False)
# net = tf.squeeze(net, axis=[1, 2])  # 去除第一、第二个维度
# logits = slim.fully_connected(net, num_outputs=class_num,
#                               activation_fn=None, scope='predict')
logits, _ = inception_resnet_v2.inception_resnet_v2(x, class_num, is_training=False)
# 计算交叉熵及其平均值
with tf.name_scope('training'):
    labels = tf.one_hot(y_, class_num)
    logits = tf.nn.softmax(logits)
    logits = tf.clip_by_value(logits, 1e-10, 1.0)
    # 优化损失函数

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(y_, tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

split_index = 0
# 初始化回话并开始训练过程。
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, './checkpoint/val_inception_resnet_model_24.ckpt')

    # 计算验证集准确率
    total = 0
    for i in range(split_index, n_val_samples, batch_size):
        print(i)
        start_pos = i
        end_pos = min(i + batch_size, n_val_samples)
        val_s1_batch = np.asarray(s1_validation[start_pos:end_pos, :, :, :])
        val_s2_batch = np.asarray(s2_validation[start_pos:end_pos, :, :, :])
        # val_X_batch = np.concatenate((val_s1_batch, val_s2_batch), axis=3)
        val_X_batch = val_s2_batch
        val_X_batch = normalize(val_X_batch)
        val_label_batch = val_label[start_pos:end_pos]

        temp = sess.run(accuracy, feed_dict={x: val_X_batch, y_: val_label_batch})
        total = total + temp * (end_pos - start_pos)


total = total / (n_val_samples - split_index)
print("验证集准确率为： " + str(total))