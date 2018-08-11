import tensorflow as tf

def fc_layer(x, shape, name, initializer=tf.keras.initializers.he_normal(), activation='relu', batch_normal=False, dropout=None):
    W = tf.get_variable('w'+str(name), shape, tf.float64, initializer)
    b = tf.get_variable('b'+str(name), shape[1], tf.float64, initializer)
    fc = tf.matmul(x, W) + b
    if activation == 'relu':
        fc = tf.nn.relu(fc)
    if activation == 'softmax':
        fc = tf.nn.softmax(fc)
    if batch_normal:
        fc = tf.layers.batch_normalization(fc)
    if dropout is not None:
        fc = tf.nn.dropout(fc, dropout)
    return fc