import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn

def network(frame1, frame2, frame3, reuse = False, scope='netflow'):

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tflearn.activations.prelu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)), \
             slim.arg_scope([slim.conv2d_transpose], activation_fn=tflearn.activations.prelu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)):

            # feature extration
            c11 = slim.conv2d(frame1, 128, [9, 9], scope='conv1_1')
            c12 = slim.conv2d(frame2, 128, [9, 9], scope='conv1_2')
            c13 = slim.conv2d(frame3, 128, [9, 9], scope='conv1_3')

            concat1_12 = tf.concat([c11, c12], 3, name='concat1_12')
            concat1_23 = tf.concat([c12, c13], 3, name='concat1_23')

            #feature merging
            c21 = slim.conv2d(concat1_12, 64, [7, 7], scope='conv2_1')
            c22 = slim.conv2d(concat1_23, 64, [7, 7], scope='conv2_2')

            # complex feature extration
            c31 = slim.conv2d(c21, 64, [3, 3], scope='conv3_1')
            c32 = slim.conv2d(c22, 64, [3, 3], scope='conv3_2')

            concat3_12 = tf.concat([c31, c32], 3, name='concat3_12')

            # non-linear mapping
            c4 = slim.conv2d(concat3_12, 32, [1, 1], scope='conv4_1')

            # residual reconstruction
            c5 = slim.conv2d(c4, 1, [5, 5], activation_fn=None, scope='conv5')

            # enhanced frame reconstruction
            output = tf.add(c5, frame2)

        return output
