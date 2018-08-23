import tensorflow as tf
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from lib import models
from db_loader import SamsungDB
from source.util import get_label
import numpy as np
import csv

batch_size = 128
steps = 5

DB = SamsungDB(mode='train')
test_DB = SamsungDB(mode='test')

x = tf.placeholder(tf.int32, [None, 16], 'input')
y = tf.placeholder(tf.int32, [None, 16], 'label')
hole = tf.placeholder(tf.float32, [None, 16], 'hole')

detail_y = []
for i in range(16):
    detail_y.append(y[:, i])

#model = models.baseline()
#model = models.baseline_dh()
model = models.baseline_embedding()

y_preds = model.predict(x)
#y_preds = tf.identity(y_preds, 'outputs') # 이름 붙이기

def get_losses(y, y_pred, hole, weight=16/3.):
    losses = []
    # hole.shape: (batch_size, 16) hole 부분만 1
    hw = hole * weight

    losses.append(sparse_ce(y[:, 0], y_pred[0], hw[:, 0])) # 주야
    losses.append(sparse_ce(y[:, 1], y_pred[1], hw[:, 1])) # 요일

    # 사망자수, 사상자수, 중상자수, 경상자수, 부상신고자수
    # 원래 5를 곱해야하는데 cls에 가중하기 위해 그냥 둠
    #losses.append(mse(tf.cast(y[:, 2:7], tf.float64), y_pred[2], hw[:, 2:7]))
    losses.append(mse(tf.cast(y[:, 2], tf.float32), y_pred[2], hw[:, 2]))
    losses.append(mse(tf.cast(y[:, 3], tf.float32), y_pred[3], hw[:, 3]))
    losses.append(mse(tf.cast(y[:, 4], tf.float32), y_pred[4], hw[:, 4]))
    losses.append(mse(tf.cast(y[:, 5], tf.float32), y_pred[5], hw[:, 5]))
    losses.append(mse(tf.cast(y[:, 6], tf.float32), y_pred[6], hw[:, 6]))

    losses.append(sparse_ce(y[:, 7], y_pred[7], hw[:, 7])) # 발생지시도
    losses.append(sparse_ce(y[:, 8], y_pred[8], hw[:, 8])) # 발생지시군구
    losses.append(sparse_ce(y[:, 9], y_pred[9], hw[:, 9])) # 사고유형_대분류
    losses.append(sparse_ce(y[:, 10], y_pred[10], hw[:, 10])) # 사고유형_중분류
    losses.append(sparse_ce(y[:, 11], y_pred[11], hw[:, 11])) # 법규위반
    losses.append(sparse_ce(y[:, 12], y_pred[12], hw[:, 12])) # 도로형태_대분류
    losses.append(sparse_ce(y[:, 13], y_pred[13], hw[:, 13])) # 도로형태
    losses.append(sparse_ce(y[:, 14], y_pred[14], hw[:, 14])) # 당사자종별_1당_대분류
    losses.append(sparse_ce(y[:, 15], y_pred[15], hw[:, 15])) # 당사자종별_2당_대분류

    return tf.reduce_mean(losses) # 평균냄

def mse(y, y_pred, weight=None):
    if weight is not None:
        return tf.reduce_mean(tf.square((y - y_pred) * weight))
    else:
        return tf.reduce_mean(tf.square(y - y_pred))

def sparse_ce(y, y_pred, weight=None):
    if weight is not None:
        return tf.losses.sparse_softmax_cross_entropy(y, y_pred, weights=weight)
    else:
        return tf.losses.sparse_softmax_cross_entropy(y, y_pred)

def rmsle(y, y_pred):
    y_true_log = tf.log(tf.maximum(y, 1e-6) + 1)
    y_pred_log = tf.log(tf.maximum(y_pred, 1e-6)+1)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred_log - y_true_log)))

losses = get_losses(y, y_preds, hole)

#optimizer = tf.train.GradientDescentOptimizer(0.01)
optimizer = tf.train.AdamOptimizer(0.001)
grad_vars = optimizer.compute_gradients(losses)
capped_grad_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grad_vars]
train_op = optimizer.apply_gradients(capped_grad_vars)
#saver = tf.train.Saver(max_to_keep=10) # 모델 저장 시 사용
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print('train_data_len:', DB.train_len())
    for step in range(steps): # step 수만큼 반복
        train_loss = 0
        val_loss = 0
        train_loss_count = 0
        val_loss_count = 0

        # train
        for idx, (train_x, train_y, train_hole) in enumerate(DB.train_generator(batch_size)):
            _, loss_ = sess.run([train_op, losses], feed_dict={x:train_x, y:train_y, hole:train_hole})
            train_loss += loss_
            train_loss_count += 1
            if idx % 1000 == 0:
                print('{}/{}, loss: {}'.format(idx, DB.train_len()/batch_size, loss_))

        # validation
        #for idx, (val_x, val_y, val_hole) in enumerate(DB.val_generator(128)):
        #    loss_ = sess.run(losses, feed_dict={x:val_x, y:val_y, hole:val_hole})
        #    val_loss += loss_
        for idx, (val_x, val_y, val_hole) in enumerate(DB.val_generator(128)):
            loss_ = sess.run(losses, feed_dict={x: val_x, y: val_y, hole: val_hole})
            val_loss += loss_
            val_loss_count += 1


        # train/val loss 출력
        print('step: {}, train_loss: {}, val_loss: {}'.format(
            step, str(train_loss/train_loss_count)[:5], str(val_loss/val_loss_count)[:5]))

        # 모델 저장 시 사용
        #saver.save(sess, "./checkpoints/baseline", global_step=step)
        #print('save lib [{}]'.format(step))

