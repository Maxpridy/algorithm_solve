import keras.layers as K_layer
import keras.models
from . import losses


def get_model():
    inputs = K_layer.Input(shape=(16,))
    dense1 = K_layer.Dense(300, activation='relu')(inputs)
    dense2 = K_layer.Dense(300, activation='relu')(dense1)

    layers = []
    layers.append(K_layer.Dense(2, activation='softmax', name='label_0')(dense2))
    layers.append(K_layer.Dense(2, activation='softmax', name='label_1')(dense2))
    layers.append(K_layer.Dense(1, name='label_2-6')(dense2))
    layers.append(K_layer.Dense(2, activation='softmax', name='label_7')(dense2))
    layers.append(K_layer.Dense(2, activation='softmax', name='label_8')(dense2))
    layers.append(K_layer.Dense(2, activation='softmax', name='label_9')(dense2))
    layers.append(K_layer.Dense(50, activation='softmax', name='label_10')(dense2))
    layers.append(K_layer.Dense(2, activation='softmax', name='label_11')(dense2))
    layers.append(K_layer.Dense(2, activation='softmax', name='label_12')(dense2))
    layers.append(K_layer.Dense(2, activation='softmax', name='label_13')(dense2))
    layers.append(K_layer.Dense(2, activation='softmax', name='label_14')(dense2))
    layers.append(K_layer.Dense(2, activation='softmax', name='label_15')(dense2))

    model = keras.models.Model(inputs=inputs, outputs=layers, name='total_model')
    model.compile(optimizer='rmsprop',
                  loss=losses.base_loss(),
                  loss_weights=[1., 1., 5., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                  metrics=['accuracy'])

    return model

