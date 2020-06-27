import numpy as np
import YUV_RGB

def init_yuv(num, hh, ww):

    y = np.zeros([num, hh, ww, 1])

    return y


def input_data(Height, Width, name):

   y = init_yuv(1, Height, Width)
   u = init_yuv(1, int(Height//2), int(Width//2))
   v = init_yuv(1, int(Height//2), int(Width//2))

   Y1, U1, V1 = YUV_RGB.yuv_import('./test_data/' + name + '.yuv', (Height, Width),1, 0)

   y[0: 1, 0:Height, 0:Width, 0] = Y1[0: 1] / 255.0
   u[0: 1, 0:int(Height//2), 0:int(Width//2), 0] = U1[0: 1] / 255.0
   v[0: 1, 0:int(Height//2), 0:int(Width//2), 0] = V1[0: 1] / 255.0

   return y, u, v

