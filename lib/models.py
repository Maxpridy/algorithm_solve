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

class baseline2():
    def __init__(self):
        print('baseline_blstm loaded')

    def predict(self, x):
        fc1 = l.fc_layer(x, [16, 300], 'fc1', activation='relu6', dropout=0.3, batch_normal=True)
        feature = l.fc_layer(fc1, [300, 300], 'fc2', dropout=0.3, activation='relu6', batch_normal=True)
        preds = []
        preds.append(l.fc_layer(feature, [300, 2], 'pred_1', activation='softmax'))
        preds.append(l.fc_layer(feature, [300, 7], 'pred_2', activation='softmax'))
        preds.append(l.fc_layer(feature, [300, 5], 'pred_37', activation=None))
        preds.append(l.fc_layer(feature, [300, 17], 'pred_8', activation='softmax'))
        preds.append(l.fc_layer(feature, [300, 209], 'pred_9', activation='softmax'))
        preds.append(l.fc_layer(feature, [300, 4], 'pred_10', activation='softmax'))
        preds.append(l.fc_layer(feature, [300, 19], 'pred_11', activation='softmax'))
        preds.append(l.fc_layer(feature, [300, 20], 'pred_12', activation='softmax'))
        preds.append(l.fc_layer(feature, [300, 9], 'pred_13', activation='softmax'))
        preds.append(l.fc_layer(feature, [300, 16], 'pred_14', activation='softmax'))
        preds.append(l.fc_layer(feature, [300, 12], 'pred_15', activation='softmax'))
        preds.append(l.fc_layer(feature, [300, 14], 'pred_16', activation='softmax'))

        return preds

class baseline_embedding():
    def __init__(self):
        print('baseline_dh loaded')

    def predict(self, x):
        embedding_size = 300
        hidden_size = 128
        embedded_features = []
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 0], 2 + 1, embedding_size, 'embed1'), 1))  # 주야 + hole
        embedded_features.append(tf.expand_dims(l.embedding(x[:, 1], 7 + 1, embedding_size, 'embed2'), 1))  # 요일 + hole

        reg_features = tf.cast(tf.tile(tf.expand_dims(x[:, 2:7], -1), [1, 1, embedding_size]), tf.float32)
        embedded_features.append(tf.expand_dims(reg_features[:, 0], 1))  # 사망자수
        embedded_features.append(tf.expand_dims(reg_features[:, 1], 1))  # 사상자수
        embedded_features.append(tf.expand_dims(reg_features[:, 2], 1))  # 중상자수
        embedded_features.append(tf.expand_dims(reg_features[:, 3], 1))  # 경상자수
        embedded_features.append(tf.expand_dims(reg_features[:, 4], 1))  # 부상신고자수

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
        blstm_features = l.blstm(embedded_features, hidden_size, 'blstm') # (batch_size, 16, embedding_size*2)
        blstm_features = tf.layers.batch_normalization(blstm_features)
        blstm_features = l.blstm(blstm_features, hidden_size, 'blstm2')  # (batch_size, 16, embedding_size*2)
        features = tf.split(blstm_features, num_or_size_splits=16, axis=1)
        final_features = []
        ff_size = 300
        for i, f in enumerate(features):
            final_features.append(l.fc_layer(tf.reshape(f, (-1, hidden_size * 2)), [hidden_size * 2, ff_size], 'refineFC_' + str(i)))

        #blstm_features = tf.layers.batch_normalization(blstm_features)
        #fc1 = l.fc_layer(tf.cast(blstm_features, tf.float32), [hidden_size*2, 200], 'fc1', activation='relu', dropout=0.5)
        #feature = l.fc_layer(fc1, [200, 200], 'fc2', dropout=0.5)
        output_size = [2, 7, 1, 1, 1, 1, 1, 17, 209, 4, 19, 20, 9, 16, 12, 14]
        preds = []
        for i, f in enumerate(final_features):
            if i > 1 and i<7: # regression (#2 ~ #7)
                preds.append(l.fc_layer(f, [ff_size, output_size[i]], 'pred_' + str(i), activation=None))
            else: # classification
                preds.append(l.fc_layer(f, [ff_size, output_size[i]], 'pred_' + str(i), activation='softmax'))

        return preds

class baseline_embedding_fc():
    def __init__(self):
        print('baseline_dh loaded')

    def predict(self, x):
        embedding_size = 128
        hidden_size = 128
        ff_size = 300
        output_size = [2, 7, 1, 1, 1, 1, 1, 17, 209, 4, 19, 20, 9, 16, 12, 14]
        embedded_features = []
        inputs = tf.split(x, num_or_size_splits=16, axis=1)
        for i, x in enumerate(inputs): # x: (batch, X)
            if i > 1 and i<7: # regression (#2 ~ #7)
                embedded_features.append(tf.cast(x, tf.float32))
            else: # classification
                embedded_features.append(tf.reshape(l.embedding(x, output_size[i] + 1, embedding_size, 'embed_' + str(i)), (-1, embedding_size)))

        embedded_features = tf.concat(embedded_features, axis=1) # (batch_size, 16, embedding_size)
        fc1 = l.fc_layer(embedded_features, [embedding_size*11 + 5, 300], 'fc1', activation='relu', batch_normal=True)
        fc2 = l.fc_layer(fc1, [300, ff_size], 'fc2', activation='relu', batch_normal=True)


        preds = []
        for i in range(16):
            if i > 1 and i<7: # regression (#2 ~ #7)
                preds.append(l.fc_layer(fc2, [ff_size, 1], 'pred_' + str(i), activation=None))
            else: # classification
                preds.append(l.fc_layer(fc2, [ff_size, output_size[i]], 'pred_' + str(i), activation='softmax'))

        return preds