import keras.layers as K_layer
import keras.models
from . import losses_keras


def get_model():
    inputs = K_layer.Input(shape=(16,))
    dense1 = K_layer.Dense(300, activation='relu')(inputs)
    dense2 = K_layer.Dense(300, activation='relu')(dense1)

    layers = []
    layers.append(K_layer.Dense(2, activation='softmax', name='label_0')(dense2))
    layers.append(K_layer.Dense(2, activation='softmax', name='label_1')(dense2))

    layers.append(K_layer.Dense(1, name='label_2')(dense2))
    layers.append(K_layer.Dense(1, name='label_3')(dense2))
    layers.append(K_layer.Dense(1, name='label_4')(dense2))
    layers.append(K_layer.Dense(1, name='label_5')(dense2))
    layers.append(K_layer.Dense(1, name='label_6')(dense2))

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
                  loss=losses_keras.base_loss(),
                  metrics=['accuracy'])

    return model

