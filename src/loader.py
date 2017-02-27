import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

from helper_functions import *

class Loader(object):
    """Load a model and test the image classification."""
    def __init__(self, model_path='/tmp/model-final.ckpt',
            test_images=None,
            test_labels=None):        
        if model_path != '/tmp/model-final.ckpt':
            curr_dir = os.getcwd()
            path = os.path.join(curr_dir,'models',model_path)
            self.model_path = path
            parameters = model_path.split('model.ckpt')[0].split('-')
            print("Parameters:", parameters)
            _, k1, kernel_size, ffn, = [int(x) for x in parameters[:4]]
        else:
            self.model_path = model_path
            k1 = 32
            kernel_size = 5
            ffn = 512
        try:
            test_images, test_labels = pickle.load(open('x_test.p','rb')), pickle.load(open('y_test.p','rb'))
        except Exception as e:
            print(e,"Data files not found.")
        tf.reset_default_graph()

        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Initialize input and target tensors
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape=[None, 784], name='x-input')
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y-input')

        # Initialize weight and bias tensors
        with tf.name_scope('convolutional_layer_1'):
            W_conv1 = weight_variable([kernel_size, kernel_size, 1, k1])
            b_conv1 = bias_variable([k1])

        with tf.name_scope('input_reshape'):
            x_image = tf.reshape(x, [-1,28,28,1])
            tf.summary.image('input', x_image, 10)

        with tf.name_scope('convolutional_layer1'):
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        with tf.name_scope('pooling_layer_1'):
            h_pool1 = max_pool_2x2(h_conv1)

        with tf.name_scope('convolutional_layer_2'):
            W_conv2 = weight_variable([kernel_size, kernel_size, k1, k1*2])
            b_conv2 = bias_variable([k1*2])        
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        with tf.name_scope('pooling_layer_2'):
            h_pool2 = max_pool_2x2(h_conv2)

        with tf.name_scope('fully_connected_layer_1'):
            W_fc1 = weight_variable([7 * 7 * k1*2, ffn])
            b_fc1 = bias_variable([ffn])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*k1*2])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                    

        with tf.name_scope('fully_connected_layer_2'):
            W_fc2 = weight_variable([ffn, 10])
            b_fc2 = bias_variable([10])
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        with tf.name_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,
                    labels=y_)
            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(diff)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
            # Initialize variables
            sess.run(init_op)
            # Restore model weights from previously saved model
            saver.restore(sess, self.model_path)
            print("Model restored from file: %s" % self.model_path)
            self.y_conv, self.x, self.y_, self.keep_prob = y_conv, x, y_, keep_prob
            self.output = sess.run([y_conv],feed_dict={x: test_images[:5].reshape(-1,784), y_: test_labels[:5], keep_prob:1.0})
            print(self.output)
            
    def classify(self,images,labels,imshow=False):
        if labels.shape[0] == 10:
            labels = [labels]            
        
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
            # Initialize variables
            sess.run(init_op)
            # Restore model weights from previously saved model
            saver.restore(sess, self.model_path)
            print("Model restored from file: %s" % self.model_path)
            self.output = sess.run([self.y_conv],feed_dict={self.x: images, self.y_: labels, self.keep_prob:1.0})
            if imshow:
                for ind, image in enumerate(images):                    
                    f, ax = plt.subplots()
                    ax.set_title('PREDICTION: ' + str(np.argmax(self.output[0][ind])) +' CORRECT: ' + str(np.argmax(labels[ind])))
                    ax.imshow(image.reshape(28,28), cmap='gray')
                    plt.axis('off')
                    plt.show()
