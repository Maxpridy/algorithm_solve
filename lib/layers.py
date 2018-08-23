import tensorflow as tf

def fc_layer(x, shape, name, initializer=tf.keras.initializers.he_normal(), activation='relu', batch_normal=False, dropout=None):
    W = tf.get_variable('w'+str(name), shape, tf.float32, initializer)
    b = tf.get_variable('b'+str(name), shape[1], tf.float32, initializer)
    fc = tf.matmul(x, W) + b
    if activation == 'relu':
        fc = tf.nn.relu(fc)
    if activation == 'relu6':
        fc = tf.nn.relu6(fc)
    if activation == 'softmax':
        fc = tf.nn.softmax(fc)
    if batch_normal:
        fc = tf.layers.batch_normalization(fc)
    if dropout is not None:
        fc = tf.nn.dropout(fc, dropout)
    return fc

def embedding(x, class_number, embedding_size, name):
    embedding_weight = tf.get_variable(name, [class_number, embedding_size])
    embedded_x = tf.nn.embedding_lookup(embedding_weight, x)  # (batch, 200, 8)
    return embedded_x

def blstm(x, n_hidden, name, keep_prob=0.4):
    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, state_is_tuple=True, name=name)
    #lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)

    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, state_is_tuple=True, name=name)
    #lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    #outputs_fw = tf.transpose(outputs[0], [1, 0, 2])
    #outputs_bw = tf.transpose(outputs[1], [1, 0, 2])
    #outputs_concat = tf.concat([outputs_fw[-1], outputs_bw[-1]], axis=1) # (batch_size, n_hidden*2)
    outputs_concat = tf.concat([outputs[0], outputs[1]], axis=-1)  # (batch_size, 16, n_hidden*2)
    #print(outputs_concat.get_shape())

    return outputs_concat