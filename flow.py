import tensorflow as tf
import tensorflow.contrib.slim as slim
import func
import tflearn

def warp_img(batch_size, imga, imgb, reuse, scope='easyflow'):

    n, h, w, c = imga.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tflearn.activations.prelu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)), \
             slim.arg_scope([slim.conv2d_transpose], activation_fn=tflearn.activations.prelu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)):

            # X4 down-scaling motion estimation
            inputs = tf.concat([imga, imgb], 3, name='flow_inp')
            c1 = slim.conv2d(inputs, 24, [5, 5], stride=2, scope='c1')
            c2 = slim.conv2d(c1, 24, [3, 3], scope='c2')
            c3 = slim.conv2d(c2, 24, [5, 5], stride=2, scope='c3')
            c4 = slim.conv2d(c3, 24, [3, 3], scope='c4')
            c5 = slim.conv2d(c4, 32, [3, 3], activation_fn=tf.nn.tanh, scope='c5')
            c5_hr = tf.reshape(c5, [n, int(h / 4), int(w / 4), 2, 4, 4])
            c5_hr = tf.transpose(c5_hr, [0, 1, 4, 2, 5, 3])
            c5_hr = tf.reshape(c5_hr, [n, h, w, 2])
            img_warp1 = func.transformer(batch_size, c, c5_hr, imgb, [h, w])

            # X2 down-scaling motion estimation
            c5_pack = tf.concat([inputs, c5_hr, img_warp1], 3, name='cat')
            s1 = slim.conv2d(c5_pack, 24, [5, 5], stride=2, scope='s1')
            s2 = slim.conv2d(s1, 24, [3, 3], scope='s2')
            s3 = slim.conv2d(s2, 24, [3, 3], scope='s3')
            s4 = slim.conv2d(s3, 24, [3, 3], scope='s4')
            s5 = slim.conv2d(s4, 8, [3, 3], activation_fn=tf.nn.tanh, scope='s5')
            s5_hr = tf.reshape(s5, [n, int(h / 2), int(w / 2), 2, 2, 2])
            s5_hr = tf.transpose(s5_hr, [0, 1, 4, 2, 5, 3])
            s5_hr = tf.reshape(s5_hr, [n, h, w, 2])
            uv = c5_hr + s5_hr
            img_warp2 = func.transformer(batch_size, c, uv, imgb, [h, w])

            # pixel-wise scale motion estimation
            s5_pack = tf.concat([inputs, uv, img_warp2], 3, name='cat2')
            a1 = slim.conv2d(s5_pack, 24, [3, 3], scope='a1')
            a2 = slim.conv2d(a1, 24, [3, 3], scope='a2')
            a3 = slim.conv2d(a2, 24, [3, 3], scope='a3')
            a4 = slim.conv2d(a3, 24, [3, 3], scope='a4')
            a5 = slim.conv2d(a4, 2, [3, 3], activation_fn=tf.nn.tanh, scope='a5')
            a5_hr = tf.reshape(a5, [n, h, w, 2, 1, 1])
            a5_hr = tf.transpose(a5_hr, [0, 1, 4, 2, 5, 3])
            a5_hr = tf.reshape(a5_hr, [n, h, w, 2])
            uv2 = a5_hr + uv

            #motion compensation (warp)
            img_warp3 = func.transformer(batch_size, c, uv2, imgb, [h, w])

    return img_warp3