import h5py
import numpy as np
import os
import tensorflow as tf
import model1
import model2
import model_low1
import model_low2
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

class_num = 17
all_num = 24119
each_class_num = [256, 1254, 2353, 849, 757, 1906, 474, 3395, 1914, 860, 2287, 382, 1202, 2747, 202, 672, 2609]
class_weight = [(1 - num / all_num) / (class_num - 1) for num in each_class_num]
# tensorflow 深度学习模型设置

weight_decay = 0.001
dropout_rate = 0.5
is_training = True
x1 = tf.placeholder(tf.float32, [None, 32, 32, 8], name="x1")
x2 = tf.placeholder(tf.float32, [None, 32, 32, 10], name="x2")
y_ = tf.placeholder(tf.int64, [None], name="y_")

# net, _ = resnet_v1.resnet_v1_26(x1)
# net = tf.squeeze(net, axis=[1, 2])  # 去除第一、第二个维度
# y1 = slim.fully_connected(net, num_outputs=class_num,
#                          activation_fn=None, scope='predict')
y1 = model1.model(x1,  is_training, dropout_rate, class_num, weight_decay)
y2 = model2.model(x2,  is_training, dropout_rate, class_num, weight_decay)
# net, _ = resnet_v1.resnet_v1_26(x1)
# net = tf.squeeze(net, axis=[1, 2])  # 去除第一、第二个维度
# logits = slim.fully_connected(net, num_outputs=class_num,
#                              activation_fn=None, scope='predict')

gamma = 2
# sen1 和 sen2的权重
sen1_weight = 0.1
# 计算交叉熵及其平均值
with tf.name_scope('training'):
    labels = tf.one_hot(y_, class_num)
    logits = tf.nn.softmax(y1 + y2)
    logits = tf.clip_by_value(logits, 1e-10, 1.0)
    # loss = -tf.reduce_mean(tf.reduce_sum(labels * tf.log(logits)))
    class_weight = np.array(class_weight)
    class_weight = class_weight.reshape([class_num, 1])
    class_weight = tf.convert_to_tensor(class_weight, tf.float32)
    class_weight_matrix = tf.matmul(labels, class_weight)

    loss = -tf.reduce_mean(tf.multiply(class_weight_matrix,
                                       tf.reduce_sum(labels * ((1 - logits) ** gamma * tf.log(logits)))))
    # 优化损失函数
    train_step = tf.train.AdamOptimizer(0.00001).minimize(loss)
    tf.summary.scalar('loss', loss)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(y_, tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

n_val_samples = s2_validation.shape[0]
train_val_y = np.argmax(label_validation, axis=1)

# 初始化回话并开始训练过程。
with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=30)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./train_model_demo', sess.graph)
    saver.restore(sess, './checkpoint/val_double_vgg_model_4.ckpt')

    for epochs in range(25):
        for i in range(0, n_val_samples, batch_size):
            start_pos = i
            end_pos = min(i + batch_size, n_val_samples)
            val_s1_batch = np.asarray(s1_validation[start_pos:end_pos, :, :, :])
            val_s2_batch = np.asarray(s2_validation[start_pos:end_pos, :, :, :])
            val_s1_batch = normalize(val_s1_batch)
            val_s2_batch = normalize(val_s2_batch)
            # val_X_batch = np.concatenate((val_s1_batch, val_s2_batch), axis=3)

            # val_X_batch = normalize(val_X_batch)
            val_label_batch = val_label[start_pos:end_pos]

            sess.run(train_step, feed_dict={x1: val_s1_batch, x2: val_s2_batch, y_: val_label_batch})
            if (i / batch_size) % 10 == 0:
                summary, l, acc = sess.run([merged, loss, accuracy], feed_dict={x1: val_s1_batch, x2: val_s2_batch,
                                                                                y_: val_label_batch})
                print("After %d training step(s), loss is %g ,accuracy is %g" % (i / batch_size, l, acc))
            train_writer.add_summary(summary, i)
        saver.save(sess, './checkpoint/val_double_vgg_model_' + str(epochs + 5) + '.ckpt')

        # 计算验证集准确率
        pred_y = []
        train_val_y = np.argmax(label_validation, axis=1)
        batch_size = 128
        n_val_samples = s2_validation.shape[0]
        total = 0
        for i in range(0, n_val_samples, batch_size):
            start_pos = i
            end_pos = min(i + batch_size, n_val_samples)
            val_s1_batch = np.asarray(s1_validation[start_pos:end_pos, :, :, :])
            val_s2_batch = np.asarray(s2_validation[start_pos:end_pos, :, :, :])
            val_s1_batch = normalize(val_s1_batch)
            val_s2_batch = normalize(val_s2_batch)
            val_label_batch = val_label[start_pos:end_pos]

            temp = sess.run(accuracy, feed_dict={x1: val_s1_batch, x2: val_s2_batch, y_: val_label_batch})
            total = total + temp * (end_pos - start_pos)

        total = total / n_val_samples
        print("第" + str(epochs) + "个epochs验证集准确率为： " + str(total))




