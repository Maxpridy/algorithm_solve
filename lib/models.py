from lib import layers as l
import tensorflow as tf

class baseline():
    def __init__(self):
        print('baseline loaded')

    def predict(self, x):
        fc1 = l.fc_layer(x, [16, 200], 'fc1', activation='relu', dropout=0.5)
        feature = l.fc_layer(fc1, [200, 200], 'fc2', dropout=0.5)
        preds = []
        preds.append(l.fc_layer(feature, [200, 2], 'pred_1', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 7], 'pred_2', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 5], 'pred_37', activation=None))
        preds.append(l.fc_layer(feature, [200, 17], 'pred_8', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 209], 'pred_9', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 4], 'pred_10', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 19], 'pred_11', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 20], 'pred_12', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 9], 'pred_13', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 16], 'pred_14', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 12], 'pred_15', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 14], 'pred_16', activation='softmax'))

        return preds


class baseline_dh():
    def __init__(self):
        print('baseline_dh loaded')

    def predict(self, x):
        embedding_size = 50
        hidden_size = 128
        embedded_features = []
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 0], 2 + 1, embedding_size, 'embed1'), 1))  # 주야 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 1], 7 + 1, embedding_size, 'embed2'), 1))  # 요일 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 2], 10 + 1, embedding_size, 'embed3'), 1))  # 사망자수(최대값) + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 3], 100 + 1, embedding_size, 'embed4'), 1))  # 사상자수(최대값) + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 4], 54 + 1, embedding_size, 'embed5'), 1))  # 중상자수(최대값) + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 5], 62 + 1, embedding_size, 'embed6'), 1))  # 경상자수(최대값) + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 6], 67 + 1, embedding_size, 'embed7'), 1))  # 부상신고자수(최대값) + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 7], 17 + 1, embedding_size, 'embed8'), 1))  # 발생지시도 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 8], 209 + 1, embedding_size, 'embed9'), 1))  # 발생지시군구 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 9], 4 + 1, embedding_size, 'embed10'), 1))  # 사고유형_대분류 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 10], 19 + 1, embedding_size, 'embed11'), 1))  # 사고유형_중분류 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 11], 20 + 1, embedding_size, 'embed12'), 1))  # 법규위반 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 12], 9 + 1, embedding_size, 'embed13'), 1))  # 도로형태_대분류 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 13], 16 + 1, embedding_size, 'embed14'), 1))  # 도로형태 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 14], 12 + 1, embedding_size, 'embed15'), 1))  # 당사자종별_1당_대분류 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 15], 14 + 1, embedding_size, 'embed16'), 1))  # 당사자종별_2당_대분류 + hole

        embedded_features = tf.concat(embedded_features, axis=1) # (batch_size, 16, embedding_size)
        blstm_features = l.blstm(embedded_features, hidden_size, 'blstm') # (batch_size, embedding_size*4)

        fc1 = l.fc_layer(tf.cast(blstm_features, tf.float64), [hidden_size*2, 200], 'fc1', activation='relu', dropout=0.5)
        feature = l.fc_layer(fc1, [200, 200], 'fc2', dropout=0.5)
        preds = []
        preds.append(l.fc_layer(feature, [200, 2], 'pred_1', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 7], 'pred_2', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 5], 'pred_37', activation=None))
        preds.append(l.fc_layer(feature, [200, 17], 'pred_8', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 209], 'pred_9', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 4], 'pred_10', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 19], 'pred_11', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 20], 'pred_12', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 9], 'pred_13', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 16], 'pred_14', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 12], 'pred_15', activation='softmax'))
        preds.append(l.fc_layer(feature, [200, 14], 'pred_16', activation='softmax'))

        return preds


class baseline_embedding():
    def __init__(self):
        print('baseline_embedding loaded')

    def predict(self, x):
        embedding_size = 50
        hidden_size = 128
        embedded_features = []
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 0], 2 + 1, embedding_size, 'embed1'), 1))  # 주야 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 1], 7 + 1, embedding_size, 'embed2'), 1))  # 요일 + hole
        #print(tf.tile(tf.expand_dims(x[:, 2], 1), (1, 50)).get_shape())
        embedded_features.append(tf.cast(tf.expand_dims(tf.tile(tf.expand_dims(x[:, 2], 1), (1, embedding_size)), 1), tf.float32)) # 사망자수(최대값) + hole
        embedded_features.append(tf.cast(tf.expand_dims(tf.tile(tf.expand_dims(x[:, 3], 1), [1, embedding_size]), 1), tf.float32)) # 사상자수(최대값) + hole
        embedded_features.append(tf.cast(tf.expand_dims(tf.tile(tf.expand_dims(x[:, 4], 1), [1, embedding_size]), 1), tf.float32)) # 중상자수(최대값) + hole
        embedded_features.append(tf.cast(tf.expand_dims(tf.tile(tf.expand_dims(x[:, 5], 1), [1, embedding_size]), 1), tf.float32)) # 경상자수(최대값) + hole
        embedded_features.append(tf.cast(tf.expand_dims(tf.tile(tf.expand_dims(x[:, 6], 1), [1, embedding_size]), 1), tf.float32)) # 부상신고자수(최대값) + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 7], 17 + 1, embedding_size, 'embed8'), 1))  # 발생지시도 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 8], 209 + 1, embedding_size, 'embed9'), 1))  # 발생지시군구 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 9], 4 + 1, embedding_size, 'embed10'), 1))  # 사고유형_대분류 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 10], 19 + 1, embedding_size, 'embed11'), 1))  # 사고유형_중분류 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 11], 20 + 1, embedding_size, 'embed12'), 1))  # 법규위반 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 12], 9 + 1, embedding_size, 'embed13'), 1))  # 도로형태_대분류 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 13], 16 + 1, embedding_size, 'embed14'), 1))  # 도로형태 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 14], 12 + 1, embedding_size, 'embed15'), 1))  # 당사자종별_1당_대분류 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 15], 14 + 1, embedding_size, 'embed16'), 1))  # 당사자종별_2당_대분류 + hole

        embedded_features = tf.concat(embedded_features, axis=1) # (batch_size, 16, embedding_size)
        print(embedded_features.get_shape())
        blstm_features = l.blstm(embedded_features, hidden_size, 'blstm') # (batch_size, 16, embedding_size*2)
        print(blstm_features.get_shape())
        blstm_features2 = l.blstm(blstm_features, hidden_size, 'blstm2')  # (batch_size, 16, embedding_size*2)
        print(blstm_features2.get_shape())

        #fc1 = l.fc_layer(tf.cast(blstm_features, tf.float64), [hidden_size*4, 200], 'fc1', activation='relu', dropout=0.5)
        #feature = l.fc_layer(fc1, [200, 200], 'fc2', dropout=0.5)
        preds = []
        preds.append(l.fc_layer(blstm_features2[:, 0, :], [hidden_size*2, 2], 'pred_1', activation='softmax'))
        preds.append(l.fc_layer(blstm_features2[:, 1, :], [hidden_size*2, 7], 'pred_2', activation='softmax'))
        preds.append(l.fc_layer(blstm_features2[:, 2, :], [hidden_size*2, 1], 'pred_3', activation=None))
        preds.append(l.fc_layer(blstm_features2[:, 3, :], [hidden_size * 2, 1], 'pred_4', activation=None))
        preds.append(l.fc_layer(blstm_features2[:, 4, :], [hidden_size * 2, 1], 'pred_5', activation=None))
        preds.append(l.fc_layer(blstm_features2[:, 5, :], [hidden_size * 2, 1], 'pred_6', activation=None))
        preds.append(l.fc_layer(blstm_features2[:, 6, :], [hidden_size * 2, 1], 'pred_7', activation=None))
        preds.append(l.fc_layer(blstm_features2[:, 7, :], [hidden_size*2, 17], 'pred_8', activation='softmax'))
        preds.append(l.fc_layer(blstm_features2[:, 8, :], [hidden_size*2, 209], 'pred_9', activation='softmax'))
        preds.append(l.fc_layer(blstm_features2[:, 9, :], [hidden_size*2, 4], 'pred_10', activation='softmax'))
        preds.append(l.fc_layer(blstm_features2[:, 10, :], [hidden_size*2, 19], 'pred_11', activation='softmax'))
        preds.append(l.fc_layer(blstm_features2[:, 11, :], [hidden_size*2, 20], 'pred_12', activation='softmax'))
        preds.append(l.fc_layer(blstm_features2[:, 12, :], [hidden_size*2, 9], 'pred_13', activation='softmax'))
        preds.append(l.fc_layer(blstm_features2[:, 13, :], [hidden_size*2, 16], 'pred_14', activation='softmax'))
        preds.append(l.fc_layer(blstm_features2[:, 14, :], [hidden_size*2, 12], 'pred_15', activation='softmax'))
        preds.append(l.fc_layer(blstm_features2[:, 15, :], [hidden_size*2, 14], 'pred_16', activation='softmax'))

        return preds