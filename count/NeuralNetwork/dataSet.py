"""Functions for downloading and reading MNIST data."""
import tensorflow as tf
import sys, traceback, time
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from ..PreProcess.readinfo import getInfo
from .. import GlobalVariables as gv

class DataSet(object):
    def __init__(self, images, labels, xlength, ylength, dtype=tf.float32):
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
          raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
        assert images.shape[0] == labels.shape[0], (
              'images.shape: %s labels.shape: %s' % (images.shape,
                                                     labels.shape))
        self._num_examples = images.shape[0]

        if dtype == tf.float32:
          # Convert from [0, 255] -> [0.0, 1.0].
          images = images.astype(np.float32)
          images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._xlength = xlength
        self._ylength = ylength

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def xlength(self):
        return self._xlength

    @property
    def ylength(self):
        return self._ylength

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def read_data_sets(instanceSize, step, label_option, mode, imageName = '', \
                   dtype=tf.float32, plot_show = 0):
    class DataSets(object):
        pass
    data_sets = DataSets()

    assert(mode == 'd-train' or mode == 'd-test' or \
           mode == 'c-train' or mode == 'c-test')

    if 'train' in mode:
        filenameList = gv.__TrainImage__
    elif 'test' in mode:
        filenameList = [imageName]
    
    if 'd' in mode:
        [images, labels, ylen, xlen] = \
            generateInstancesNN(instanceSize, step, label_option, \
                            filenameList, plot_show)
        data_sets = DataSet(images, labels, \
                            xlen, ylen, dtype=dtype)
    elif 'c' in mode:
        [images, labels, ylen, xlen] = \
            generateInstancesNN(instanceSize, step, label_option, \
                            filenameList, plot_show)
        data_sets = DataSet(images, labels, \
                            xlen, ylen, dtype=dtype)

    return data_sets

def generateInstancesNN(instanceSize, step, label_option, \
                        filenameList, plot_show = 1):

    trainInstances = np.array([])
    trainLabels    = np.array([])
    xlen = 0
    ylen = 0

    for trainImage in filenameList:
        print 'Generating training instances from [' + trainImage + \
            ']... { instanceSize: ' + str(instanceSize) + ' step: ' + \
            str(step) + ' }'
        train_file = gv.__DIR__ + gv.__TrainImageDir__ + trainImage
        train = io.imread(train_file)

        image_files, bubble_num, bubble_regions = getInfo()
        positiveLabels = bubble_regions[image_files.index(trainImage)]

        [m,n,c] = train.shape

        Y = np.arange(np.floor(instanceSize/2), \
                      (m - np.ceil(instanceSize/2)), step)
        X = np.arange(np.floor(instanceSize/2), \
                      (n - np.ceil(instanceSize/2)), step)
    
        totalInstancesY = len(Y)
        totalInstancesX = len(X)
        ylen = totalInstancesY
        xlen = totalInstancesX

        progressRatePrevious = 0

        Instances = np.zeros( (instanceSize ** 2 * c, \
                               totalInstancesX * totalInstancesY ))
        Labels = np.zeros((label_option, totalInstancesX * totalInstancesY))

        ind = -1
    
        start_time = time.time()
        for i in Y:
            for j in X:
                currentY = i
                currentX = j        
       
                boundaryT = currentY - np.floor(instanceSize/2)
                boundaryD = currentY + np.floor(instanceSize/2)
                boundaryL = currentX - np.floor(instanceSize/2)
                boundaryR = currentX + np.floor(instanceSize/2)
        
                ind = ind + 1
        
                temp = train[boundaryT : boundaryD, boundaryL : boundaryR, :]

                Instances[:, ind] = np.reshape(temp, \
                                               (instanceSize ** 2 * c, 1)).T
                
 		if(label_option == 2):
                   factor = 3;
                else: 
                   factor = 2;
            	probabiliyIndex = max(0,np.floor(np.sqrt(\
                	gaussian2d(currentY, currentX, positiveLabels, \
                	scale = 0.3))*label_option*factor)-1);
            	probabiliyIndex = min(label_option-1, probabiliyIndex);# ???
            	Labels[probabiliyIndex, ind] = 1;
    
            elapse_time = time.time() - start_time
            progressRate = np.true_divide(ind, \
                               totalInstancesX*totalInstancesY) * 100
            remain_time = np.true_divide(elapse_time, progressRate) * 100 \
                          - elapse_time
            progressRate = np.floor(np.true_divide(progressRate, 10))
            if (progressRate > progressRatePrevious):
                print str(progressRate*10) + \
                    '% train instances created, remaining time: ' +\
                    time.strftime(" %H:%M:%S", time.gmtime(remain_time))
            progressRatePrevious = progressRate
        if trainInstances.size == 0:
            trainInstances = Instances.T
            trainLabels = Labels.T
        else:
            trainInstances = np.append(trainInstances, Instances.T, axis=0)
            trainLabels = np.append(trainLabels, Labels.T, axis=0)

        if(plot_show == 1):
            fig, ax = plt.subplots(2)
            ax[0].imshow(train)
            ax[0].set_title('Original Image')
            ax[1].imshow(np.reshape(np.sum( \
                np.multiply(Labels.T,np.arange(label_option)).T,0),\
                (totalInstancesY,totalInstancesX)))
            ax[1].set_title('Labels')
            plt.show()

    return [trainInstances, trainLabels, ylen, xlen]

def gaussian2d(y, x, positiveLabels, scale = 0.2):

    x_c = positiveLabels[:,0] + np.ceil(np.true_divide(positiveLabels[:,2],2))
    y_c = positiveLabels[:,1] + np.ceil(np.true_divide(positiveLabels[:,3],2))
    a = np.ceil(np.true_divide(positiveLabels[:,2],2)) * scale
    b = np.ceil(np.true_divide(positiveLabels[:,3],2)) * scale

    A = 100*np.true_divide(1, 2*np.pi*a*b)
#   A = 1
    powerX = np.true_divide(np.power(np.subtract(x,x_c),2) \
            , np.multiply(2, np.power(a,2))) 
    powerY = np.true_divide(np.power(np.subtract(y,y_c),2) \
            , np.multiply(2, np.power(b,2))) 
    
    Pr = np.multiply(A, np.exp(-(powerX+powerY)))
    return np.sum(Pr)

def main():
     try:
         instanceSize = 20;
         step = 10;
         edge = 4;
         scale =10;
	 label_option = 2;
         read_data_sets(instanceSize, step, label_option, 'train', \
                        plot_show = 1)
         read_data_sets(instanceSize, step, label_option, 'test',\
                        'detector_1_no_5_angle_3.jpg', plot_show = 1)
     except KeyboardInterrupt:
         print "Shutdown requested... exiting"
     except Exception:
         traceback.print_exc(file=sys.stdout)
     sys.exit(0)
 
if __name__ == '__main__':
     main()

