import numpy as np
from numpy import *
from scipy.misc import imresize
import YUV_RGB
from skimage import io
import matplotlib.pyplot as plt

def show_img(Enhanced_Y, non_PQF_Frame_96_Y, non_PQF_Frame_96_U, non_PQF_Frame_96_V, Height, Width):

    Enhanced_R, Enhanced_G, Enhanced_B = YUV_RGB.yuv2rgb(Enhanced_Y[0, :, :, 0],
                                                         non_PQF_Frame_96_U[0, :, :, 0],
                                                         non_PQF_Frame_96_V[0, :, :, 0], Height, Width)

    HEVC_R, HEVC_G, HEVC_B = YUV_RGB.yuv2rgb(non_PQF_Frame_96_Y[0, :, :, 0],
                                             non_PQF_Frame_96_U[0, :, :, 0],
                                             non_PQF_Frame_96_V[0, :, :, 0], Height, Width)

    Enhanced_frame = np.zeros([Height, Width, 3])
    HEVC_frame = np.zeros([Height, Width, 3])
    Enhanced_ball = np.zeros([150, 150, 3])
    HEVC_ball = np.zeros([150, 150, 3])

    Enhanced_frame[0:Height, 0:Width, 0] = Enhanced_R / 255.0
    Enhanced_frame[0:Height, 0:Width, 1] = Enhanced_G / 255.0
    Enhanced_frame[0:Height, 0:Width, 2] = Enhanced_B / 255.0

    io.imsave('Frame_96_our_MFQE.bmp', Enhanced_frame)

    Enhanced_frame[115: 119, 73: 74 + 30, 0] = 0
    Enhanced_frame[115: 119, 73: 74 + 30, 1] = 0
    Enhanced_frame[115: 119, 73: 74 + 30, 2] = 1.0

    Enhanced_frame[115 + 30: 119 + 30, 73: 74 + 30, 0] = 0
    Enhanced_frame[115 + 30: 119 + 30, 73: 74 + 30, 1] = 0
    Enhanced_frame[115 + 30: 119 + 30, 73: 74 + 30, 2] = 1.0

    Enhanced_frame[115: 119 + 30, 73: 77, 0] = 0
    Enhanced_frame[115: 119 + 30, 73: 77, 1] = 0
    Enhanced_frame[115: 119 + 30, 73: 77, 2] = 1.0

    Enhanced_frame[115: 119 + 30, 73 + 30: 77 + 30, 0] = 0
    Enhanced_frame[115: 119 + 30, 73 + 30: 77 + 30, 1] = 0
    Enhanced_frame[115: 119 + 30, 73 + 30: 77 + 30, 2] = 1.0

    HEVC_frame[0:Height, 0:Width, 0] = HEVC_R / 255.0
    HEVC_frame[0:Height, 0:Width, 1] = HEVC_G / 255.0
    HEVC_frame[0:Height, 0:Width, 2] = HEVC_B / 255.0

    io.imsave('Frame_96_HEVC.bmp', HEVC_frame)

    HEVC_frame[115: 119, 73: 74 + 30, 0] = 0
    HEVC_frame[115: 119, 73: 74 + 30, 1] = 0
    HEVC_frame[115: 119, 73: 74 + 30, 2] = 1.0

    HEVC_frame[115 + 30: 119 + 30, 73: 74 + 30, 0] = 0
    HEVC_frame[115 + 30: 119 + 30, 73: 74 + 30, 1] = 0
    HEVC_frame[115 + 30: 119 + 30, 73: 74 + 30, 2] = 1.0

    HEVC_frame[115: 119 + 30, 73: 77, 0] = 0
    HEVC_frame[115: 119 + 30, 73: 77, 1] = 0
    HEVC_frame[115: 119 + 30, 73: 77, 2] = 1.0

    HEVC_frame[115: 119 + 30, 73 + 30: 77 + 30, 0] = 0
    HEVC_frame[115: 119 + 30, 73 + 30: 77 + 30, 1] = 0
    HEVC_frame[115: 119 + 30, 73 + 30: 77 + 30, 2] = 1.0

    Enhanced_ball_R = imresize(Enhanced_R[117: 147, 75: 105] / 255.0, [150, 150], 'bilinear', mode='F')
    Enhanced_ball_G = imresize(Enhanced_G[117: 147, 75: 105] / 255.0, [150, 150], 'bilinear', mode='F')
    Enhanced_ball_B = imresize(Enhanced_B[117: 147, 75: 105] / 255.0, [150, 150], 'bilinear', mode='F')

    HEVC_ball_R = imresize(HEVC_R[117: 147, 75: 105] / 255.0, [150, 150], 'bilinear', mode='F')
    HEVC_ball_G = imresize(HEVC_G[117: 147, 75: 105] / 255.0, [150, 150], 'bilinear', mode='F')
    HEVC_ball_B = imresize(HEVC_B[117: 147, 75: 105] / 255.0, [150, 150], 'bilinear', mode='F')

    Enhanced_ball[0:150, 0:150, 0] = Enhanced_ball_R
    Enhanced_ball[0:150, 0:150, 1] = Enhanced_ball_G
    Enhanced_ball[0:150, 0:150, 2] = Enhanced_ball_B

    HEVC_ball[0:150, 0:150, 0] = HEVC_ball_R
    HEVC_ball[0:150, 0:150, 1] = HEVC_ball_G
    HEVC_ball[0:150, 0:150, 2] = HEVC_ball_B

    plt.subplot(221);
    plt.title('HEVC baseline');
    plt.imshow(HEVC_frame);
    plt.axis('off')
    plt.subplot(222);
    plt.title('Our MFQE approach');
    plt.imshow(Enhanced_frame);
    plt.axis('off')
    plt.subplot(223);
    plt.imshow(HEVC_ball);
    plt.axis('off')
    plt.subplot(224);
    plt.imshow(Enhanced_ball);
    plt.axis('off')

    plt.show()

    # return HEVC_frame, Enhanced_frame, HEVC_ball, Enhanced_ball