import numpy as np
import flow
import tensorflow as tf
import os
import net
import data
import show
import matplotlib

os.environ['CUDA_VISIBLE_DEVICES']='0'

#matplotlib.use('TkAgg')
#Uncomment the above line if plt.show() does not work

Height = 240
Width = 416
Channel = 1
batch_size = 1

#Session
config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config = config)

#Placeholder
x1 = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
x2 = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
x3 = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])

## MC-subnet
x1to2 = flow.warp_img(batch_size, x2, x1, False)
x3to2 = flow.warp_img(batch_size, x2, x3, True)

## QE-subnet
x2_enhanced = net.network(x1to2, x2, x3to2)

##Import data
PQF_Frame_93_Y,  PQF_Frame_93_U, PQF_Frame_93_V = data.input_data(Height, Width, 'Frame_93')
non_PQF_Frame_96_Y,  non_PQF_Frame_96_U, non_PQF_Frame_96_V = data.input_data(Height, Width, 'Frame_96')
PQF_Frame_97_Y,  PQF_Frame_97_U, PQF_Frame_97_V = data.input_data(Height, Width, 'Frame_97')

##Load model
saver = tf.train.Saver()
saver.restore(sess, './HEVC_QP37_model/model.ckpt')

##Run test
Enhanced_Y = sess.run(x2_enhanced, feed_dict={x1: PQF_Frame_93_Y[0:1,0:Height, 0:Width, 0:1],
                                              x2: non_PQF_Frame_96_Y[0:1,0:Height, 0:Width, 0:1],
                                              x3: PQF_Frame_97_Y[0:1,0:Height, 0:Width, 0:1]})

##Show results
# HEVC_frame, Enhanced_frame, HEVC_ball, Enhanced_ball = \
show.show_img(Enhanced_Y, non_PQF_Frame_96_Y, non_PQF_Frame_96_U, non_PQF_Frame_96_V, Height, Width)

