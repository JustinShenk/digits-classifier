import os, sys
from os.path import join, realpath, dirname
from time import time
import colorsys
import pickle
from prettytable import PrettyTable

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import gaussian_filter

import matplotlib
import matplotlib.image as mpimg
matplotlib.use('nbAgg')
import matplotlib.pyplot as plt

import tensorflow as tf

from PIL import Image

from scipy.ndimage import filters
import pickle

def rgbtogray(img):
    return 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]

class Stamp_Digits(object):
    
    RAW_ROW_P = 42
    RAW_COL_P = 34
    R_Pixel = 28
    C_Pixel = 28
    DIMS = R_Pixel*C_Pixel
    NUMBERS = 10
    HIGH_CUT = 7
    SIDE_CUT = 3

    def __init__(self, directory):
        number_data, number_targets, number_count = self.load_data(directory)

        self.data = number_data
        self.targets = number_targets
        self.number_counts = number_count
        self.train_idx = None
        self.test_idx = None

    def load_data(self, directory=os.getcwd()):

        number_data = np.empty((0,Stamp_Digits.DIMS))
        number_targets = np.empty((0,))
        number_count = np.zeros((Stamp_Digits.NUMBERS))
        total_count = 0

        all_files = []
        # go through all files in all subdirectories
        for path, subdirs, files in os.walk(directory):
            for name in files:
                filepath = join(path,name)
                all_files.append(filepath)

        all_files.sort()
        for filepath in all_files:
            digit = int(filepath[-5])
            # load image using matplotlib
            bmp_rgb = mpimg.imread(filepath)
            # convert to grayscale image
            bmp_gray = rgbtogray(bmp_rgb)
            # get dimensions
            r, c = np.shape(bmp_gray)
            # get amount of images for this number
            img_num = np.uint32(c/Stamp_Digits.RAW_COL_P)    
            number_count[digit] += img_num
            
            # enlarge number data
            number_data = np.concatenate((number_data,np.zeros((img_num,Stamp_Digits.DIMS))),axis=0)
            
            # get the images as row vectors
            for i in range(0, img_num):
                number_data[i + total_count,:] = bmp_gray[Stamp_Digits.HIGH_CUT:Stamp_Digits.HIGH_CUT+Stamp_Digits.R_Pixel,
                i*Stamp_Digits.RAW_COL_P+Stamp_Digits.SIDE_CUT:i*Stamp_Digits.RAW_COL_P+Stamp_Digits.SIDE_CUT+Stamp_Digits.C_Pixel].reshape(1,Stamp_Digits.DIMS)
            
            number_targets = np.concatenate((number_targets,np.zeros((img_num,))),axis=0)
            number_targets[total_count:total_count+img_num] = digit
            total_count += img_num

            number_data = number_data.astype(np.uint8)

        return number_data, number_targets, number_count

    def gen_train_test_split(self, test_size=0.2):
        '''Generates the indices for the training and the testing sets

        @param  test_size: The fraction of the data supposed to be testing
        @type   test_size: C{float}
        '''
        n_data = len(self.data)
        self.train_idx, self.test_idx = train_test_split(np.arange(n_data),test_size=test_size,random_state=42)

    def get_validation_data(self):
        return self.data[self.test_idx], self.targets[self.test_idx].T

    def get_test_data(self):
        return self.data[self.test_idx], self.targets[self.test_idx].T

    def get_random_sample(self, batch_size):
        if(batch_size>len(self.train_idx)):
            batch_size = len(self.train_idx)
        sample_idx = np.random.permutation(self.train_idx)[0:batch_size]
        train_data = self.data[sample_idx]
        train_lables = self.targets[sample_idx]
        return train_data, train_lables.T

    def get_next_batch(self, batch_size, iteration):
        if(batch_size>len(self.train_idx)):
            batch_size = len(self.train_idx)


        b_per_epoch = int(len(self.train_idx)/batch_size)
        batch_num = iteration % b_per_epoch
        if(batch_num == 0):
            self.train_rand_perm = np.random.permutation(self.train_idx)

        ofset = batch_num*batch_size
        sample_idx = self.train_rand_perm[ofset:ofset+batch_size]
        train_data = self.data[sample_idx]
        train_lables = self.targets[sample_idx]
        return train_data, train_lables.T

# ----------- Enhancement -------------------

    def contrast_enhancement(self):

        start_idx = 0
        n_i = 5
        n_e = 4
        digit_image= np.empty((0,n_i*3*Stamp_Digits.C_Pixel))

        M = self._get_pca_sorted_eig_vectors()

        for next_digit in self.number_counts:
            digit_row = np.empty((n_e*Stamp_Digits.R_Pixel,0))
            qualities = [start_idx, start_idx+int(next_digit/2), int(start_idx+next_digit-n_i)]
            for img_quality in qualities:
                for img in range(0,n_i):
                    digit = self.data[img_quality+img]

                    res_digit = np.reshape(digit,(Stamp_Digits.R_Pixel, Stamp_Digits.C_Pixel))
                    res_digit = res_digit.astype(int)
                    # res_digit = (res_digit-np.min(res_digit))/(np.max(res_digit)-np.min(res_digit))*255
                    # res_digit = (res_digit+0.5).astype(int)

                    all_images = np.concatenate((res_digit,
                        self._get_nonlinear_global_enhancement(res_digit),
                        self._get_entropy_enhanced_image(res_digit),
                        self._get_laplace_image(res_digit)),axis=0)
                    digit_row = np.append(digit_row,all_images,axis=1)

            digit_image = np.append(digit_image,digit_row,axis=0)
            start_idx+=int(next_digit)

        img = Image.fromarray(np.uint8(digit_image))
        img.save("../results/enhancement_channel1.bmp")

    def pca_images(self):
        start_idx = 0
        n_i = 5 
        n_p = 5
        digit_image= np.empty((0,n_i*3*Stamp_Digits.C_Pixel))

        M = self._get_pca_sorted_eig_vectors()

        for next_digit in self.number_counts:
            digit_row = np.empty((n_p*Stamp_Digits.R_Pixel,0))
            qualities = [start_idx, start_idx+int(next_digit/2), int(start_idx+next_digit-n_i)]
            for img_quality in qualities:
                for img in range(0,n_i):
                    digit = self.data[img_quality+img]
                    img_max = np.max(digit)
                    img_min = np.min(digit)
                    img_dif = img_max-img_min

                    res_digit = np.reshape(digit,(Stamp_Digits.R_Pixel, Stamp_Digits.C_Pixel))
                    res_digit = res_digit.astype(int)
                    res_digit = (res_digit-np.min(res_digit))/(np.max(res_digit)-np.min(res_digit))*img_dif+img_min
                    res_digit = (res_digit+0.5).astype(int)

                    all_images = np.concatenate((res_digit,
                        self._get_pca_img(M,digit,30),
                        self._get_pca_img(M,digit,100),
                        self._get_pca_img(M,digit,500),
                        self._get_pca_img(M,digit,1000)),axis=0)
                    digit_row = np.append(digit_row,all_images,axis=1)

            digit_image = np.append(digit_image,digit_row,axis=0)
            start_idx+=int(next_digit)

        img = Image.fromarray(np.uint8(digit_image))
        img.save("../results/pca_channel1.bmp")

    def create_shifted_images(self):
        start_idx = 0
        n_i = 5 
        digit_image= np.empty((0,n_i*3*Stamp_Digits.C_Pixel))

        for next_digit in self.number_counts:
            digit_row = np.empty((n_i*Stamp_Digits.R_Pixel,0))
            qualities = [start_idx, start_idx+int(next_digit/2), int(start_idx+next_digit-n_i)]
            for img_quality in qualities:
                for img in range(0,n_i):
                    digit = self.data[img_quality+img]

                    res_digit = np.reshape(digit,(Stamp_Digits.R_Pixel, Stamp_Digits.C_Pixel))
                    res_digit = res_digit.astype(int)
                    # res_digit = (res_digit-np.min(res_digit))/(np.max(res_digit)-np.min(res_digit))*255
                    # res_digit = (res_digit+0.5).astype(int)

                    all_images = np.concatenate((res_digit,
                        self._randomly_shift_image(res_digit),
                        self._randomly_shift_image(res_digit),
                        self._randomly_shift_image(res_digit),
                        self._randomly_shift_image(res_digit)),axis=0)
                    digit_row = np.append(digit_row,all_images,axis=1)

            digit_image = np.append(digit_image,digit_row,axis=0)
            start_idx+=int(next_digit)

        img = Image.fromarray(np.uint8(digit_image))
        img.save("../results/img_shift_channel1.bmp")

    def create_partially_occluded_images(self):
        start_idx = 0
        n_i = 5 
        digit_image= np.empty((0,n_i*3*Stamp_Digits.C_Pixel))

        for next_digit in self.number_counts:
            digit_row = np.empty((n_i*Stamp_Digits.R_Pixel,0))
            qualities = [start_idx, start_idx+int(next_digit/2), int(start_idx+next_digit-n_i)]
            for img_quality in qualities:
                for img in range(0,n_i):
                    digit = self.data[img_quality+img]

                    res_digit = np.reshape(digit,(Stamp_Digits.R_Pixel, Stamp_Digits.C_Pixel))
                    res_digit = res_digit.astype(int)
                    # res_digit = (res_digit-np.min(res_digit))/(np.max(res_digit)-np.min(res_digit))*255
                    # res_digit = (res_digit+0.5).astype(int)

                    all_images = np.concatenate((res_digit,
                        self._randomly_occlude_image(res_digit),
                        self._randomly_occlude_image(res_digit),
                        self._randomly_occlude_image(res_digit),
                        self._randomly_occlude_image(res_digit)),axis=0)
                    digit_row = np.append(digit_row,all_images,axis=1)

            digit_image = np.append(digit_image,digit_row,axis=0)
            start_idx+=int(next_digit)

        img = Image.fromarray(np.uint8(digit_image))
        img.save("../results/img_occluded_channel1.bmp")

    def create_partially_overlaying_images(self):
        pass

    def _get_nonlinear_global_enhancement(self, img):
        max_color = np.max(img)
        nonlinear_global_enhancement = 255 * np.power(img/max_color, 2)
        nonlinear_global_enhancement = (nonlinear_global_enhancement+0.5).astype(int)
        return nonlinear_global_enhancement

    def _get_entropy_enhanced_image(self, img):
        hist = np.bincount(img.ravel())

        max_color = np.max(img)
        perc = hist/Stamp_Digits.DIMS
        acc = 0
        val_dict = {}

        for i,val in enumerate(perc):
            acc += val
            val_dict[i] = int(acc*max_color+0.5)

        entropy_enhanced_image = np.zeros(np.shape(img))
        for i in range(len(img)):
            for j in range(len(img[0])):
                entropy_enhanced_image[i,j] = val_dict[img[i,j]]

        entropy_enhanced_image = entropy_enhanced_image.astype(int)

        return entropy_enhanced_image

    def _get_laplace_image(self, img):
        gauss_filter3 = 1/16*np.array([[1,2,1],[2,4,2],[1,2,1]])
        gauss_filter5 = 1/256*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
        gauss_filter7 = np.outer([1,6,15,20,10,6,1],[1,6,15,20,10,6,1])
        gauss_filter7 = 1/np.sum(gauss_filter7)*gauss_filter7
        smoothed_image = filters.convolve(img,gauss_filter7)
        laplace_image = filters.laplace(smoothed_image)
        laplace_image = (laplace_image-np.min(laplace_image))/(np.max(laplace_image)-np.min(laplace_image))*255
        laplace_image = (laplace_image+0.5).astype(int)
        return laplace_image

    def _get_pca_sorted_eig_vectors(self):
        number_data = self.data
        (n, d) = number_data.shape;
        number_data = number_data - np.tile(np.mean(number_data, 0), (n, 1));
        cov = np.dot(number_data.T, number_data)/(n-1)
        # create a list of pair (eigenvalue, eigenvector) tuples
        eig_val, eig_vec = np.linalg.eig(cov)
        eig_val_sum = np.sum(np.real(eig_val))
        eig_pairs = []
        for x in range(0,len(eig_val)):
            eig_pairs.insert(x, (np.abs(eig_val[x])/eig_val_sum,  np.real(eig_vec[:,x])))
        
        # sort the list starting with the highest eigenvalue
        eig_pairs.sort(key=lambda tup: tup[0], reverse=True)
        M = np.hstack((eig_pairs[i][1].reshape(d,1) for i in range(0,len(eig_pairs))))
        return M

    def _get_pca_img(self, pcs, img, num_pcs):
        img_trans = np.dot(img,pcs)
        img_res = np.dot(pcs[:,0:num_pcs],img_trans[0:num_pcs].T)
        img_res = img_res.reshape(Stamp_Digits.R_Pixel,Stamp_Digits.C_Pixel)
        img_res = (img_res-np.min(img_res))/(np.max(img_res)-np.min(img_res))*255
        img_res = (img_res+0.5).astype(int)
        return img_res

    def _randomly_shift_image(self, img):
        shift_r = np.random.randint(-5,5)
        shift_c = np.random.randint(-5,5)
        shift_img = np.zeros(np.shape(img))
        a = shift(img, [shift_r,shift_c], mode='nearest')
        return a

    def _randomly_occlude_image(self, img):
        side = np.random.randint(1,5)
        portion = np.random.rand()*0.5+0.1
        keep = 1-portion
        dist_img = np.copy(img)
        mean_background = 0

        hist = np.bincount(img.ravel())
        perc = hist/Stamp_Digits.DIMS
        acc = 0
        count = 0
        for i,val in enumerate(perc):
            acc += val
            if(acc>=0.1):
                if(acc>=0.9):
                    break
            mean_background += i*hist[i]
            count += hist[i]
        mean_background/=count
        
        # right
        if(side==1):
            dist_part =  dist_img[:,int(Stamp_Digits.C_Pixel*keep):]
            noise = np.random.normal(0,2,np.shape(dist_part))
            dist_img[:,int(Stamp_Digits.C_Pixel*keep):] = mean_background+noise
        # left
        elif(side==2):
            dist_part =  dist_img[:,0:int(Stamp_Digits.C_Pixel*portion)]
            noise = np.random.normal(0,2,np.shape(dist_part))
            dist_img[:,0:int(Stamp_Digits.C_Pixel*portion)] = mean_background+noise
        # bottom
        elif(side==3):
            dist_part =  dist_img[int(Stamp_Digits.R_Pixel*keep):,:]
            noise = np.random.normal(0,2,np.shape(dist_part))
            dist_img[int(Stamp_Digits.R_Pixel*keep):,:] = mean_background+noise
        # top
        else:
            dist_part =  dist_img[0:int(Stamp_Digits.R_Pixel*portion),:]
            noise = np.random.normal(0,2,np.shape(dist_part))
            dist_img[0:int(Stamp_Digits.R_Pixel*portion),:] = mean_background+noise

        return dist_img.astype(int)


class Conv_Net(object):

    def __init__(self, data):
        self.data = data

    def setup_conv_net(self, model_name, conv_layers=[[5,5,1,32],[5,5,32,64]], max_pool=[True,True], feedforward_layers=[[7 * 7 * 64, 512]]):
        self.model_name = model_name
        # Initialize input and target tensors.
        self.x = tf.placeholder(tf.float32, shape=[None, Stamp_Digits.R_Pixel, Stamp_Digits.C_Pixel, 1])
        self.y = tf.placeholder(tf.int64, shape=[None])
        self.e = tf.placeholder(tf.float32, None)
        self.keep_prob = tf.placeholder(tf.float32)

        # define the input for the next layer
        conv_input = self.x
        # set up all the conv layers
        for c_layer, do_pool in zip(conv_layers,max_pool):
            # set the weights
            W_conv = self.weight_variable(c_layer)
            # set the biases
            b_conv = self.bias_variable([c_layer[-1]])

            # create the conv layer with relu neurons
            h_conv = tf.nn.relu(self.conv2d(conv_input, W_conv) + b_conv)

            # add a 2x2 pooling layer if requested
            if(do_pool):
                h_pool = self.max_pool_2x2(h_conv)
                # input for the next layer is then the output of the pooling layer
                conv_input = h_pool
            else:
            # otherwise the output of the conv_net is the input for the next layer
                conv_input = h_conv

        # conv layers are done. need to flatten the output for the final feed foward layer
        ff_input = tf.reshape(conv_input, [-1, feedforward_layers[0][0]])

        # set up all the feed forward layers
        for f_layer in feedforward_layers:
            # set the weights
            W_ff = self.weight_variable(f_layer)
            # set the biases
            b_ff = self.bias_variable([f_layer[-1]])

            # create the forward layer with a relu neuron
            pre_dropout = tf.nn.relu(tf.matmul(ff_input, W_ff) + b_ff)
            # add a dropout to prevent overfitting (dropout is adjustable)
            # output is the new input for the next layer
            ff_input = tf.nn.dropout(pre_dropout, self.keep_prob)

        # create the last layer linked to the output neurons
        # set up weights
        W_out_layer = self.weight_variable([feedforward_layers[-1][-1], 10])
        # set up biases
        b_out_layer = self.bias_variable([10])

        # create the output function
        self.y_conv = tf.matmul(ff_input, W_out_layer) + b_out_layer

        # create crossentropy function
        self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.y_conv, self.y))

        # create training step function
        self.train_step = tf.train.AdamOptimizer(self.e).minimize(self.cross_entropy)

        # create accuracy function
        self.accuracy = tf.equal(tf.argmax(tf.nn.softmax(self.y_conv), 1), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

    def train_net(self, save_path, training_steps=30000, batchsize=300, learning_rate=1e-3,):

        try:
            # plotting 
            plotStepSize = 25
            trainingAccurcies = np.ones(training_steps)
            validationAccuracies = np.ones(training_steps)

            trainingCrossEntropies = np.zeros(training_steps)
            validationCrossEntropies = np.zeros(training_steps)

            accFig, accAx = plt.subplots(1,1)
            accFig.suptitle("accuracy",fontsize=20)
            dpi = accFig.get_dpi()
            accFig.set_size_inches(1920.0/float(dpi),1080.0/float(dpi))
            ceFig, ceAx = plt.subplots(1,1)
            ceFig.suptitle("cross_entropy",fontsize=20)
            dpi = ceFig.get_dpi()
            ceFig.set_size_inches(1920.0/float(dpi),1080.0/float(dpi))
            # plt.show(block=False)
            # actFig, actAx = plt.subplots(10,1)

            # we will save the best model. this is the minimum performanced for a model to be saved
            best_acc = 0.999
            bench_time = 0

            # start session
            session = tf.Session()
            # initialize variables
            session.run(tf.global_variables_initializer())
            # get a saver instance
            saver = tf.train.Saver()
            
            s_time = time()
            for step in range(training_steps):

                # choose learning rate depending on the progess
                if step < 250:
                    learning_rate = 1e-3
                elif step < 350:
                    learning_rate = 8e-4
                elif step < 450:
                    learning_rate = 6e-4
                elif step < 550:
                    learning_rate = 2e-4
                else:
                    learning_rate = 1e-4

                # get new training samples
                images, labels = self.data.get_next_batch(batchsize,step)
                # get the data in the right shape
                images = images.reshape([-1, Stamp_Digits.R_Pixel, Stamp_Digits.C_Pixel, 1])
                # train the network and get crossentropy and accuracy scores for the plot
                trainingAccurcies[step], trainingCrossEntropies[step], _ = session.run([self.accuracy, self.cross_entropy, self.train_step], feed_dict= {self.x: images,self.y: labels, self.e:learning_rate, self.keep_prob: 0.5})
                
                # every few steps confirm the progess on a validation set
                if step % plotStepSize == 0 or step == 0 or step == training_steps-1:
                    # get validation data
                    images, labels = self.data.get_validation_data()
                    # reshape the data
                    images = images.reshape([-1, Stamp_Digits.R_Pixel, Stamp_Digits.C_Pixel, 1])
                    # get crossentropy and accuracy scores for the validation set
                    validationAccuracy, validationcrossEntropy, outputActivation = session.run([self.accuracy, self.cross_entropy, self.y_conv], feed_dict= {self.x: images, self.y: labels, self.e:learning_rate,  self.keep_prob: 1})

                    # if the model is the currently best one evaluated at the validation set save it
                    if(validationAccuracy>best_acc):
                        bench_time = time() - s_time
                        best_acc = validationAccuracy
                        # log the progress
                        print("New best accuracy with %.5f"% (best_acc,))
                        # save the model
                        saved = saver.save(session, "./models/" + self.model_name + ".chkp")

                    # if the model was sufficiently trained and we recognize a hugh performace drop (2%) its pretty certain we ran into overfitting, so we might stop
                    if(best_acc>0.999 and validationAccuracy<(best_acc-0.02)):
                        print("Training stopped after %d iterations!"%(step,))
                        print("Accuracy dropped from %.5f to %.5f"%(best_acc,validationAccuracy))
                        break

                    # add the progess of the validation set to the plot
                    if step != training_steps-1:
                        validationAccuracies[step:step+plotStepSize] = [validationAccuracy] * plotStepSize
                        validationCrossEntropies[step:step+plotStepSize] = [validationcrossEntropy] * plotStepSize
                    
                    # draw the accuracy progress
                    accAx.cla()
                    accAx.plot(trainingAccurcies, color = 'b')
                    accAx.plot(validationAccuracies, color = 'r')
                    accFig.canvas.draw()
                    
                    # draw the crossentropy progress
                    ceAx.cla()
                    ceAx.plot(trainingCrossEntropies, color = 'b')
                    ceAx.plot(validationCrossEntropies, color = 'r')
                    ceFig.canvas.draw()

                    # log the step every 100 steps
                    if(step%100==0):
                        print("%d iterations done!"%(step,))

                    # outputActivation = outputActivation.T
                    # for i, ax in enumerate(actAx):
                    #     ax.cla()
                    #     ax.set_xticklabels([])
                    #     ax.set_yticklabels([])
                    #     ax.matshow(np.matrix(outputActivation[i]), extent=[0, 6000, 0, 1], aspect='auto', cmap = plt.get_cmap('Blues'))
                    # actFig.canvas.draw()

            images, labels = self.data.get_validation_data()
            images = images.reshape([-1, Stamp_Digits.R_Pixel, Stamp_Digits.C_Pixel, 1])
            t_start = time()
            testAccuracy = session.run(self.accuracy, feed_dict= {self.x: images, self.y: labels, self.keep_prob: 1})
            t_end = time()

            return bench_time, best_acc, validationAccuracies, t_end-t_start

        finally:    
            # get the final accuracy
            images, labels = self.data.get_test_data()
            images = images.reshape([-1, Stamp_Digits.R_Pixel, Stamp_Digits.C_Pixel, 1])
            testAccuracy = session.run(self.accuracy, feed_dict= {self.x: images, self.y: labels, self.keep_prob: 1})

            # save the final plots
            accFig.savefig("../results/" + self.model_name + "_acc.png")
            ceFig.savefig("../results/" + self.model_name + "_ce.png")

            plt.close()
            plt.close()
            
            # 
            print("Final: ", testAccuracy)
            session.close()

    def predict(self, model):
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # saver = tf.train.import_meta_graph('./models/resnet.chkp.meta')
            saver.restore(session, model)

            images, labels = self.data.get_test_data()
            images = images.reshape([-1, Stamp_Digits.R_Pixel, Stamp_Digits.C_Pixel, 1])

            # preds = session.run(self.y_conv, feed_dict= {self.x: images, self.y: labels, self.keep_prob: 1})
            # pred = np.argmax(preds,axis=1)

            # for i in range(len(labels)):
            #     if(pred[i]!=labels[i]):
            #         print("pred: " + str(pred[i]) + " label: " + str(labels[i]))
            #         plt.imshow(images[i].reshape(Stamp_Digits.R_Pixel,Stamp_Digits.C_Pixel),cmap=plt.get_cmap('gray'))
            #         plt.show()

            t_start = time()
            testAccuracy = session.run(self.accuracy, feed_dict= {self.x: images, self.y: labels, self.keep_prob: 1})
            t_end = time()
            print("Final: ", testAccuracy)
            return (t_start-t_end)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(value=0.1, shape=shape)
        return tf.Variable(initial)
        
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def getColors(color_num):
    colors=[]
    for i in np.arange(0., 360., 360. / color_num):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

def explore_different_models(digit_data,training_steps=10000,batchsize=256):
    conv_net = Conv_Net(data=digit_data)

    time_predict = np.zeros(2*3*4)
    time_99_9_accuracy = np.zeros(2*3*4)
    best_accuracy = np.zeros(2*3*4)
    accuracy_chart = [None]*(2*3*4)
    setup=0

    colors = getColors(2*3*4)
    fig, ax = plt.subplots(1,1)
    fig.suptitle("kickass", fontsize=20)
    dpi = fig.get_dpi()
    fig.set_size_inches(1920.0/float(dpi),1080.0/float(dpi))

#    figManager = plt.get_current_fig_manager()
#    figManager.resize(*figManager.window.maxsize())

    header = ["model_name", "best_acc", "mean_acc_last_2k", "std_acc_last_2k", "time_0.999acc", "prediction_time"]
    table = PrettyTable(header)

    for k_s in [5,3]:
        for k_n in [32,16]:
            for ff_s in [512,256,128]:
                label = "ks" + (str(k_s) + "_kn" + str(k_n) + "_fn" + str(ff_s))
                conv_net.setup_conv_net(model_name=label,
                    conv_layers=[[k_s,k_s,1,k_n],[k_s,k_s,k_n,k_n*2]],
                    max_pool=[True,True],
                    feedforward_layers=[[7 * 7 * k_n*2, ff_s]])
                path = "./models/" + label + ".chkp"
                time_99_9_accuracy[setup], best_accuracy[setup], accuracy_chart[setup], time_predict[setup]= conv_net.train_net(save_path=path, training_steps=training_steps, batchsize=batchsize)

                last_accs = np.array(accuracy_chart[setup])[int(training_steps*0.9):]
                table.add_row([label,
                    "%.5f"%(best_accuracy[setup],),
                    "%.5f"%(np.mean(last_accs),),
                    "%.5f"%(np.std(last_accs),),
                    "%.2f"%(time_99_9_accuracy[setup]/60,),
                    "%.5f"%(time_predict[setup],)])

                ax.plot(accuracy_chart[setup], color = colors[setup], label = label)
                legend = ax.legend(loc='lower right',ncol=4, shadow=True)
                setup+=1

    with open("../results/kickass_table.txt", 'w') as f:
        f.write(table.get_string())

    legend = ax.legend(loc='lower right',ncol=4, shadow=True)
    fig.savefig("../results/kickass.png")

def explore_further_models(digit_data,training_steps=10000,batchsize=256):
    models = [("best_ks3_kn32_fn256",[[3,3,1,32],[3,3,32,64]],[True,True],[[7*7*64,256]]), # best previous setup
    ("swapped_kernel_counts_64_32",[[3,3,1,64],[3,3,64,32]],[True,True],[[7*7*32,256]]), # swap kernel counts
    ("mixed_kernel_sizes_5_3",[[5,5,1,32],[3,3,32,64]],[True,True],[[7*7*64,256]]), # mix of kernel size
    ("one_conv_layer_64",[[3,3,1,64]],[True],[[14*14*64,256]]), # one conv layer only
    ("one_conv_layer_128",[[3,3,1,128]],[True],[[14*14*128,256]]), # one larger only
    ("three_conv_layer_32",[[3,3,1,32],[3,3,32,64],[3,3,64,32]],[True,True,True],[[4*4*32,256]]), # three conv layers
    ("three_conv_layer_16",[[3,3,1,16],[3,3,16,32],[3,3,32,16]],[True,True,True],[[4*4*16,256]]), # three conv layers smaller
    ("two_ff_layers",[[3,3,1,32],[3,3,32,64]],[True,True],[[7*7*64,256],[256,128]]), # 2 ff layers
    ("no_pooling_2nd_conv",[[3,3,1,32],[3,3,32,64]],[True,False],[[14*14*64,256]]), # second no pooling layer
    ("no_pooling_1st_conv",[[3,3,1,32],[3,3,32,64]],[False,True],[[14*14*64,256]])] # first no pooling layer

    conv_net = Conv_Net(data=digit_data)


    time_predict = np.zeros(2*3*4)
    time_99_9_accuracy = np.zeros(2*3*4)
    best_accuracy = np.zeros(2*3*4)
    accuracy_chart = [None]*(2*3*4)
    setup=0

    colors = getColors(2*3*4)
    fig, ax = plt.subplots(1,1)
    fig.suptitle("kickass", fontsize=20)
    dpi = fig.get_dpi()
    fig.set_size_inches(1920.0/float(dpi),1080.0/float(dpi))

 #   figManager = plt.get_current_fig_manager()
 #   figManager.resize(*figManager.window.maxsize())

    header = ["model_name", "best_acc", "mean_acc_last_2k", "std_acc_last_2k", "time_0.999acc", "prediction_time"]
    table = PrettyTable(header)

    for model in models:

        label = model[0]
        conv_net.setup_conv_net(model_name=label,
            conv_layers=model[1],
            max_pool=model[2],
            feedforward_layers=model[3])
        path = "./models/" + label + ".chkp"
        time_99_9_accuracy[setup], best_accuracy[setup], accuracy_chart[setup], time_predict[setup]= conv_net.train_net(save_path=path, training_steps=training_steps, batchsize=batchsize)

        last_accs = np.array(accuracy_chart[setup])[int(training_steps*0.9):]
        table.add_row([label,
            "%.5f"%(best_accuracy[setup],),
            "%.5f"%(np.mean(last_accs),),
            "%.5f"%(np.std(last_accs),),
            "%.2f"%(time_99_9_accuracy[setup]/60,),
            "%.5f"%(time_predict[setup],)])

        ax.plot(accuracy_chart[setup], color = colors[setup], label = label)
        legend = ax.legend(loc='lower right', shadow=True, ncol=4)
        setup+=1

    with open("../results/kickass_harder_table.txt", 'w') as f:
        f.write(table.get_string())

    legend = ax.legend(loc='lower right', shadow=True, ncol=4)
    fig.savefig("../results/kickass_harder.png")


if __name__ == "__main__":
    # digit_data = Stamp_Digits("../data")
    # pickle.dump(digit_data, open("../data.p","wb"))

    digit_data = pickle.load(open("../data_p2.p","rb"))
    digit_data.gen_train_test_split(test_size=0.2)



    # digit_data.gen_train_test_split(test_size=0.2)
    # digit_data.create_shifted_images()
    # digit_data.create_partially_occluded_images()
    # digit_data.pca_images()
    # digit_data.contrast_enhancement()

    # conv_net = Conv_Net(data=digit_data)
    # conv_net.setup_conv_net(conv_layers=[[5,5,1,32],[5,5,32,64]], feedforward_layers=[[7 * 7 * 64, 512]])
    # conv_net.train_net(save_path="./models/bla.chkp", training_steps=100000, batchsize=300)

    # conv_net_check = Conv_Net(data=digit_data)
    # conv_net_check.setup_conv_net(conv_layers=[[5,5,1,32],[5,5,32,64]], feedforward_layers=[[7 * 7 * 64, 512]])
    # conv_net_check.predict(model="./models/resnet.chkp")

    # explore_different_models(digit_data,15000,256)
    explore_further_models(digit_data,15000,256)
