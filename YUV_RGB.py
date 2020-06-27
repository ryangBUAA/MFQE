import numpy as np
from numpy import *
from scipy.misc import imresize

def yuv_import(filename, dims, numfrm, startfrm):
    fp = open(filename, 'rb')
    blk_size = np.prod(dims) * 3 / 2
    fp.seek(int(blk_size * startfrm), 0)

    d00 = dims[0] // 2
    d01 = dims[1] // 2

    Y = np.zeros((numfrm, dims[0], dims[1]), np.uint8, 'C')
    U = np.zeros((numfrm, d00, d01), np.uint8, 'C')
    V = np.zeros((numfrm, d00, d01), np.uint8, 'C')


    for i in range(numfrm):
        for m in range(dims[0]):
            for n in range(dims[1]):
                Y[i, m, n] = ord(fp.read(1))
        for m in range(d00):
            for n in range(d01):
                U[i, m, n] = ord(fp.read(1))
        for m in range(d00):
            for n in range(d01):
                V[i, m, n] = ord(fp.read(1))

    fp.close()
    return (Y, U, V)

def yuv2rgb(Y, U, V, height, width,):

    U = imresize(U, [height, width], 'bilinear', mode = 'F')
    V = imresize(V, [height, width], 'bilinear', mode = 'F')
    Y = Y * 255.0

    rf = Y + 1.4075 * (V * 255.0 - 128.0)
    gf = Y - 0.3455 * (U * 255.0 - 128.0) - 0.7169 * (V * 255.0 - 128.0)
    bf = Y + 1.7790 * (U * 255.0 - 128.0)

    for m in range(height):
        for n in range(width):

            if(rf[m, n] > 255):
                rf[m, n] = 255

            if(gf[m, n] > 255):
                gf[m, n] = 255

            if(bf[m, n] > 255):
                bf[m, n] = 255

            if (rf[m, n] < 0):
                rf[m, n] = 0

            if (gf[m, n] < 0):
                gf[m, n] = 0

            if (bf[m, n] < 0):
                bf[m, n] = 0

    r = rf.astype(uint8)
    g = gf.astype(uint8)
    b = bf.astype(uint8)

    return r, g, b

