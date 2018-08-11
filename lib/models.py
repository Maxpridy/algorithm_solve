from lib import layers as l

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