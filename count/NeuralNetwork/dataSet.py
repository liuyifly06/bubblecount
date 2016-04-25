"""Functions for downloading and reading MNIST data."""
import tensorflow as tf
import sys, traceback, time, warnings
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from ..PreProcess.readinfo import getInfo
from .. import GlobalVariables as gv
from ..ProgressBar import progress

class DataSet(object):
    def __init__(self, images, labels, xlength, ylength, dtype=tf.float32):
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32'
                            % dtype)
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
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

def read_data_sets(instanceSize, step, numOfClasses, instanceMode, 
                   labelMode, imageName = '', dtype=tf.float32,
                   plot_show = 0):
    class DataSets(object):
        pass
    data_sets = DataSets()

    assert(instanceMode == 'train' or instanceMode == 'test')
    if 'train' in instanceMode:
        filenameList = gv.__TrainImage__ 
    elif 'test' in instanceMode:
        filenameList = [imageName]

    [images, labels, ylen, xlen] = generateInstancesNN(instanceSize, 
                step, numOfClasses, filenameList, labelMode, plot_show)
    data_sets = DataSet(images, labels, xlen, ylen, dtype=dtype)
    return data_sets

def generateInstancesNN(instanceSize, step, numOfClasses, filenameList, 
                        mode, plot_show = 1):

    allInstances = np.array([])
    allLabels    = np.array([])
    xlen = 0
    ylen = 0

    image_files, bubble_num, bubble_regions = getInfo()

    for i, imageFilename in enumerate(filenameList):
        print ('Generating instances from [' + imageFilename + 
               ']... { patch width: ' + str(instanceSize) + ' step: ' + 
               str(step) + ' }')

        filename = gv.__DIR__ + gv.__TrainImageDir__ + imageFilename
        imageData = io.imread(filename)
        
        positiveLabels = bubble_regions[image_files.index(imageFilename)]

        [m,n,c] = imageData.shape

        Y = np.arange(np.floor(instanceSize/2), 
                      (m - np.ceil(instanceSize/2)), step)
        X = np.arange(np.floor(instanceSize/2),
                      (n - np.ceil(instanceSize/2)), step)
    
        ylen = len(Y)
        xlen = len(X)

        image_show = np.zeros((ylen, xlen))
        
        if(i <= 1):
            print ('memory cost for one image is '+
                   str(instanceSize**2*c*xlen*ylen*4) +
                   ' bytes')

        instances = np.zeros((instanceSize ** 2 * c, xlen * ylen ))
        labels = np.zeros((numOfClasses, xlen * ylen))

        ind = -1    
        start_time = time.time()

        PROGRESS = progress.progress(0, xlen*ylen,
            prefix_info = imageFilename,
            suffix_info = str(i+1)+'/'+str(len(filenameList)) )

        for iy, currentY in enumerate(Y):
            for ix, currentX in enumerate(X):      
       
                boundaryT = currentY - np.floor(instanceSize/2)
                boundaryD = currentY + np.floor(instanceSize/2)
                boundaryL = currentX - np.floor(instanceSize/2)
                boundaryR = currentX + np.floor(instanceSize/2)

                ind = ind + 1
                temp = imageData[boundaryT:boundaryD, boundaryL:boundaryR, :]
                instances[:, ind] = np.reshape(temp,(instanceSize**2*c, 1)).T
                
                assert (mode == 'NUM' or mode == 'PRO')
                if  (mode == 'PRO'):
                    probabiliyIndex = gaussian2d(currentY, currentX,
                    positiveLabels, scale = 0.3) * (numOfClasses/2 + 2)
                    probabiliyIndex = min(numOfClasses-1, probabiliyIndex)
                    image_show[iy, ix] = int(probabiliyIndex)               
                    labels[probabiliyIndex, ind] = 1
                elif(mode == 'NUM'):
                    box = [boundaryT, boundaryD, boundaryL, boundaryR]
                    bubbleNums = numberOfBubbles(box, positiveLabels)
                    bubbleNums = bubbleNums * max(1, numOfClasses/10)
                    bubbleNums = min(numOfClasses-1, bubbleNums)
                    image_show[iy, ix] = np.around(bubbleNums)                
            	    labels[int(bubbleNums), ind] = 1;
                PROGRESS.setCurrentIteration(ix + iy*xlen + 1)
                PROGRESS.printProgress()

        # append current instance data to all instance data
        if allInstances.size == 0:
            allInstances = instances.T
            allLabels = labels.T
        else:
            allInstances = np.append(allInstances, instances.T, axis=0)
            allLabels = np.append(allLabels, labels.T, axis=0)

        if(plot_show == 1):
            fig, ax = plt.subplots(2)
            ax[0].imshow(imageData)
            ax[0].set_title('Original Image')
            img = ax[1].imshow(image_show)
            ax[1].set_title('Labels')
            plt.colorbar(img)
            plt.show()

    return [allInstances, allLabels, ylen, xlen]

def gaussian2d(y, x, positiveLabels, scale = 0.2):
    x_c = positiveLabels[:,0] + np.ceil(np.true_divide(positiveLabels[:,2], 2))
    y_c = positiveLabels[:,1] + np.ceil(np.true_divide(positiveLabels[:,3], 2))
    a = np.ceil(np.true_divide(positiveLabels[:,2], 2)) * scale
    b = np.ceil(np.true_divide(positiveLabels[:,3], 2)) * scale

    A = 100*np.true_divide(1, 2*np.pi*np.multiply(a, b))

    powerX = np.true_divide(np.power(np.subtract(x,x_c),2),
                            np.multiply(2, np.power(a,2))) 
    powerY = np.true_divide(np.power(np.subtract(y,y_c),2),
                            np.multiply(2, np.power(b,2)))
    
    Pr = np.multiply(A, np.exp(-(powerX+powerY)))
    return np.sum(Pr)

def numberOfBubbles(box, positiveLabels):
    bubbleNum = 0

    area = np.multiply(positiveLabels[:,2], positiveLabels[:,3])

    region = np.array([positiveLabels[:,1],
                       positiveLabels[:,1] + positiveLabels[:,3],
                       positiveLabels[:,0],
                       positiveLabels[:,0] + positiveLabels[:,2]]).T

    index = np.all([np.logical_not(np.any(
           [region[:,0] >= box[1], region[:,1] <= box[0]], axis = 0)),
                    np.logical_not(np.any(
           [region[:,2] >= box[3], region[:,3] <= box[2]], axis = 0))],
           axis = 0)
    
    if(np.sum(index)==0):
        return 0
    edge = np.ones((np.sum(index), 4))
    
    edge[:,0] = np.absolute(np.subtract(region[index, 0], box[1]))
    edge[:,1] = np.absolute(np.subtract(region[index, 1], box[0]))
    edge[:,2] = box[1] - box[0]
    edge[:,3] = region[index, 1] - region[index, 0]

    Yedge = np.amin(edge, axis = 1)

    edge[:,0] = np.absolute(np.subtract(region[index, 2], box[3]))
    edge[:,1] = np.absolute(np.subtract(region[index, 3], box[2]))
    edge[:,2] = box[3] - box[2]
    edge[:,3] = region[index, 3] - region[index, 2]

    Xedge = np.amin(edge, axis = 1)
    
    bubbleIncludes = np.true_divide(np.multiply(Xedge, Yedge), area[index])   
    return np.sum(bubbleIncludes)

def intervalOverlap(interval1, interval2):
    if(np.any([interval1[0] >= interval2[1], interval1[1] <= interval2[0]])):
        return 0
    else:
        return min(interval1[1], interval2[1]) - max(interval1[0], interval2[0])

def main():
     try:
         instanceSize = 50;
         step = 5;
         edge = 4;
         scale =10;
	 numOfClasses = 100;
         read_data_sets(instanceSize, step, numOfClasses, 'train', 'PRO',
                        plot_show = 1)
         read_data_sets(instanceSize, step, numOfClasses, 'test',  'PRO',
                        'detector_1_no_5_angle_3.jpg', plot_show = 1)
         read_data_sets(instanceSize, step, numOfClasses, 'train', 'NUM',
                        plot_show = 1)
         read_data_sets(instanceSize, step, numOfClasses, 'test',  'NUM',
                        'detector_1_no_5_angle_3.jpg', plot_show = 1)
     except KeyboardInterrupt:
         print "Shutdown requested... exiting"
     except Exception:
         traceback.print_exc(file=sys.stdout)
     sys.exit(0)
 
if __name__ == '__main__':
     main()

