import os, sys, time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from os.path import join, realpath, dirname
from scipy.ndimage.filters import gaussian_laplace
from PIL import Image
from scipy.ndimage import imread

C_PIXELS = 34
CAMERAS = [1,2,4] # `k1`, `k2`, `k4` only
FOLDERS = [3,4] # `HART_3`, `HART_4` only
ORIENTATION = [0] # `o0` only
DIR = '../TrainingBMPs/LS3_HART_'
K_FILENAME = ['BSP_k','_o0_sym_']

def prepare_images():
    data = {}
    data['original'] = []
    data['gaussian'] = []
    data['targets'] = []
    
    # for folder in FOLDERS:
    imgs = []
    targets = []
    gradient_images = []
    for folder in [3, 4]: # Temporary override
        directory = DIR + str(folder)
        for camera in CAMERAS:
            for number in range(10):
                path = join(directory,str(camera).join(K_FILENAME) + str(number) + '.bmp')
                image = imread(path, mode='L')

                for j in range(0,len(image[0])-1,C_PIXELS):
                    # Crop images.
                    img = image[7:35,j+3:j+31] # 28 x 28 pixels.
                    imgs.append(img)
                    targets.append(number)
                    # Get second derivative with Laplacian Gaussian.
                    laplacian = gaussian_laplace(img,sigma=2)
                    gradient_images.append(laplacian)
                    
    # Add images to data dictionary.
    data['original'] = imgs
    data['gaussian'] = gradient_images
    data['targets'] = targets
    return data

def preview_imgs(data, n = 20, gaussian = False):
    folder = 'gaussian' if gaussian else 'original'
    for ind,item in enumerate(data[folder]):
        if ind % n == 0:
            plt.title(str(data['targets'][ind]))
            plt.imshow(item,cmap='gray')
            plt.show()
                
def save_pickle(data, path='data.p'):
    pickle.dump(data, open(path,'wb'))

def load_pickle(path='data.p'):
    return pickle.load(open(path, 'rb'))

def collect_images(data, filters='original'):
    x = []
    y = []
    for index,img in enumerate(data[filters]):
        x.append(img.astype(float))
        y.append(data['targets'][index])
    x = np.array(x)
    y = np.array(y)
    return x,y

def weight_variable(shape):
    with tf.name_scope('weights'):
        initial = tf.truncated_normal(shape, stddev=0.1)
        weight = tf.Variable(initial)
        variable_summaries(weight)
        return weight

def bias_variable(shape):
    with tf.name_scope('biases'):
        initial = tf.constant(0.1, shape=shape)
        bias = tf.Variable(initial)
        variable_summaries(bias)
        return bias
    
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

