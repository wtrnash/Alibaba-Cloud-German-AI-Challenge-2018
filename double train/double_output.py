import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import pandas as pd
import tensorflow as tf
import model_low2
import model_low1
import model1
import model2
import resnet_v1
import tensorflow.contrib.slim as slim
import pre_trained_resnet
import vgg


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
path_test = os.path.join(base_dir, "round2_test_b_20190211.h5")

fid_test = h5py.File(path_test, 'r')

print("-" * 60)
print("test part")
s1_test = fid_test['sen1']
print(s1_test.shape)
s2_test = fid_test['sen2']
print(s2_test.shape)

class_num = 17
# sen1 和 sen2的权重
sen1_weight = 0.3
# 声明占位符
x1 = tf.placeholder(tf.float32, [None, 32, 32, 8], name="x1")
x2 = tf.placeholder(tf.float32, [None, 32, 32, 10], name="x2")

"""
with slim.arg_scope(pre_trained_resnet.resnet_arg_scope(0.0)):
    net, _ = pre_trained_resnet.resnet_v1_50(x1)
    net = tf.squeeze(net, axis=[1, 2])  # 去除第一、第二个维度
    logits1 = slim.fully_connected(net, num_outputs=class_num,
                              activation_fn=None, scope='predict')

with slim.arg_scope(vgg.vgg_arg_scope(0.0)):
    logits2, _ = vgg.vgg_16(x2, class_num, False, 1.0)
"""
# net, _ = resnet_v1.resnet_v1_26(x1, is_training=False)
# net = tf.squeeze(net, axis=[1, 2])  # 去除第一、第二个维度
# logits1 = slim.fully_connected(net, num_outputs=class_num,
#                             activation_fn=None, scope='predict')

logits1 = model1.model(x1, False, 0.0, class_num, 0.0)
logits2 = model2.model(x2, False, 0.0, class_num, 0.0)

logits = tf.nn.softmax(logits1 + logits2)
logits = tf.clip_by_value(logits, 1e-10, 1)
result = tf.argmax(logits, 1)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, './checkpoint/new_val_double_vgg_0.769/val_double_vgg_model_26.ckpt')

n_test_samples = s2_test.shape[0]
batch_size = 200
result_list = []
for i in range(0, n_test_samples, batch_size):
    start_pos = i
    end_pos = min(i + batch_size, n_test_samples)
    test_s1_batch = np.asarray(s1_test[start_pos:end_pos, :, :, :])
    test_s2_batch = np.asarray(s2_test[start_pos:end_pos, :, :, :])
    test_s1_batch = normalize(test_s1_batch)
    test_s2_batch = normalize(test_s2_batch)
    value = sess.run(result, feed_dict={x1: test_s1_batch, x2: test_s2_batch})  # 传入网络得到结果
    for j in range(end_pos - start_pos):
        temp = value[j]
        result_list.append(temp)

result_list = transform_one_hot(result_list)
final_list = []
for i in range(len(result_list)):
    temp = []
    for j in range(class_num):
        temp.append(int(result_list[i][j]))

    final_list.append(temp)

result_df = pd.DataFrame(final_list)  # 转为DataFrame
result_df.to_csv("./submit/0.769_val_double_submit.csv", encoding='utf-8', index=None, header=None)
