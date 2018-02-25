import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score


def top_k_features(adj_m, fea_m, k, scope):
    adj_m = tf.expand_dims(adj_m, axis=1, name=scope+'/expand1')
    fea_m = tf.expand_dims(fea_m, axis=-1, name=scope+'/expand2')
    feas = tf.multiply(adj_m, fea_m, name=scope+'/mul')
    feas = tf.transpose(feas, perm=[2, 1, 0], name=scope+'/trans1')
    top_k = tf.nn.top_k(feas, k=k, name=scope+'/top_k').values
    #pre, post = tf.split(top_k, 2, axis=2, name=scope+'/split')
    top_k = tf.concat([fea_m, top_k], axis=2, name=scope+'/concat')
    top_k = tf.transpose(top_k, perm=[0, 2, 1], name=scope+'/trans2')
    return top_k


def simple_conv(adj_m, outs, num_out, adj_keep_r, keep_r, is_train, scope,
                act_fn=tf.nn.elu, norm=True, **kw):
    adj_m = dropout(adj_m, adj_keep_r, is_train, scope+'/drop1')
    outs = dropout(outs, keep_r, is_train, scope+'/drop2')
    outs = fully_connected(outs, num_out, scope+'/fully', None)
    outs = tf.matmul(adj_m, outs, name=scope+'/matmul')
    #if norm:
    #    outs = batch_norm(outs, is_train, scope=scope+'/norm', act_fn=None)
    outs = outs if not act_fn else act_fn(outs, scope+'/act')
    return outs


def graph_conv(adj_m, outs, num_out, adj_keep_r, keep_r, is_train, scope, k=5,
               act_fn=tf.nn.relu6, **kw):
    num_in = outs.shape[-1].value
    adj_m = dropout(adj_m, adj_keep_r, is_train, scope+'/drop1')
    outs = top_k_features(adj_m, outs, k, scope+'/top_k')
    outs = dropout(outs, keep_r, is_train, scope+'/drop1')
    outs = conv1d(outs, (num_in+num_out)//2, (k+1)//2+1, scope+'/conv1', None, True)
    outs = act_fn(outs, scope+'act1') if act_fn else outs
    outs = dropout(outs, keep_r, is_train, scope+'/drop2')
    outs = conv1d(outs, num_out, k//2+1, scope+'/conv2', None)
    outs = tf.squeeze(outs, axis=[1], name=scope+'/squeeze')
    return batch_norm(outs, True, scope+'/norm2', act_fn)


def fully_connected(outs, dim, scope, act_fn=tf.nn.elu):
    outs = tf.contrib.layers.fully_connected(
        outs, dim, activation_fn=None, scope=scope+'/dense',
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        biases_initializer=tf.contrib.layers.xavier_initializer())
        #weights_initializer=tf.random_normal_initializer(),
        #biases_initializer=tf.random_normal_initializer())
    return act_fn(outs, scope+'/act') if act_fn else outs


def conv1d(outs, num_out, k, scope, act_fn=tf.nn.relu6, use_bias=False):
    l2_func = tf.contrib.layers.l2_regularizer(5e-4, scope)
    outs = tf.layers.conv1d(
        outs, num_out, k, activation=act_fn, name=scope+'/conv',
        padding='valid', use_bias=use_bias,
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    return outs


def chan_conv(adj_m, outs, num_out, keep_r, is_train, scope,
              act_fn=tf.nn.relu6):
    outs = dropout(outs, keep_r, is_train, scope)
    outs = tf.matmul(adj_m, outs, name=scope+'/matmul')
    in_length = outs.shape.as_list()[-1]
    outs = tf.expand_dims(outs, axis=-1, name=scope+'/expand')
    kernel = in_length - num_out + 1
    outs = conv1d(outs, 1, kernel, scope+'/conv', act_fn)
    outs = tf.squeeze(outs, axis=[-1], name=scope+'/squeeze')
    return batch_norm(outs, True, scope, act_fn)


def dropout(outs, keep_r, is_train, scope):
    if keep_r < 1.0:
        return tf.contrib.layers.dropout(
            outs, keep_r, is_training=is_train, scope=scope)
    return outs


def batch_norm(outs, is_train, scope, act_fn=tf.nn.relu6):
    return tf.contrib.layers.batch_norm(
        outs, scale=True,
        activation_fn=act_fn, fused=True,
        is_training=is_train, scope=scope,
        updates_collections=None)


def masked_softmax_cross_entropy(preds, labels, mask, name='loss'):
    with tf.variable_scope(name):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask, name='accuracy'):
    with tf.variable_scope(name):
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)


def score(y_true, y_pred):
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.round().astype(np.int32)
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(f1_score(y_true[:,i], y_pred[:,i], average="micro"))
    #return max(scores)
    return sum(scores) / len(scores)
    #return scores/y_true.shape[1]
