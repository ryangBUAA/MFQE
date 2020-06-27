Our latest work of video enhancement:

Ren Yang, Xiaoyan Sun, Mai Xu and Wenjun Zeng, "Quality-Gated Convolutional LSTM for Enhancing Compressed Video", in *IEEE International Conference on Multimedia and Expo (ICME)*, 2019.

is avaiable at https://github.com/ryangchn/QG-ConvLSTM.git (codes included).

# MFQE Test Demo 

The demo includes the trained model (for HEVC sequences at QP = 37) and test codes of the MF-CNN in our MFQE approach. 

In this demo, we use frame 96 (non-PQF) of the video sequence BasketballPass as an example. 

As a result, the non-PQF (frame 96) can be enhanced by our MFQE approach, taking advantage of the adjacent PQFs (frames 93 and 97).

Run "main.py" to run the demo. 

After runing, the frame 96 compressed by HEVC and enhanced by our MFQE approach are shown as "Frame_96_HEVC.bmp" and "Frame_96_our_MFQE.bmp", respectively.


# Recommended settings

Ubuntu 14.04, Tensorflow 1.3.0, Python 2.7

# Dependency

Tensorflow, TFLearn, Numpy, Scipy, matplotlib, skimage

# Contact

E-mail: r.yangchn@gmail.com

WeChat: yangren93
