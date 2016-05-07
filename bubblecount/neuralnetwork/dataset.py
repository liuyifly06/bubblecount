import tensorflow as tf
import numpy as np
import bubblecount.globalvar as gv
from matplotlib import pyplot as plt
from skimage import io
from bubblecount.preprocess.readinfo import getinfo

class DataSet(object):
    def __init__(self, instances, labels, xlength, ylength, imagedata, stride,
                 dtype=tf.float32):

        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
          raise TypeError('Invalid image dtype %r, expected uint8 or float32'
                            % dtype)
        assert instances.shape[0] == labels.shape[0], (
          'instances.shape: %s labels.shape: %s' % (instances.shape,
                                                    labels.shape))
        self._num_examples = instances.shape[0]

        if dtype == tf.float32:
          for i in range(len(imagedata)):
            imagedata[i] = imagedata[i].astype(np.float32)
            imagedata[i] = np.multiply(imagedata[i], 1.0 / 255.0)

        self._instances        = instances
        self._labels           = labels
        self._epochs_completed = 0
        self._index_in_epoch   = 0
        self._xlength          = xlength
        self._ylength          = ylength
        self._imagedata        = imagedata
        self._stride           = stride

    @property
    def imagedata(self):
        return self._imagedata
    
    @property
    def instances(self):
        return self._instances

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

    @property
    def images(self):
        index = np.ones((self._instances.shape[0],
                         self._instances.shape[1]), dtype=bool)
        return self.extractInstances(index)

    def extractInstances(self, index):
        trainable = []
        allinstances = self._instances[index]
        
        ncols = self._instances.shape[1]
        nrows = np.size(allinstances)/ncols

        allinstances = np.reshape(allinstances, (nrows, ncols))
        for i in range(allinstances.shape[0]):
            instance = allinstances[i].astype(int)
            current  = self._imagedata[instance[2]]
            trainable.append(
              np.reshape(current[instance[1]:(instance[1]+self._stride),
                                 instance[0]:(instance[0]+self._stride)],
                         (self._stride**2*3)
              )
            )
        return np.array(trainable)
        
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        index = np.zeros((self._instances.shape[0],
                          self._instances.shape[1]), dtype=bool)
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._instances = self._instances[perm]
            self._labels    = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        index[start:end] = True
        return self.extractInstances(index), self._labels[start:end]

def read_data_sets(instanceSize, step, numOfClasses, instanceMode, 
                   labelMode, imageName = '', dtype = tf.float32,
                   plot_show = 0):
    class DataSets(object):
        pass
    data_sets = DataSets()

    assert(instanceMode == 'train' or instanceMode == 'test')
    if 'train' in instanceMode:
        filenameList = gv.__TrainImage__ 
    elif 'test' in instanceMode:
        filenameList = [imageName]

    [instances, labels, ylen, xlen, imagedata] = generateInstancesNN(
        instanceSize, step, numOfClasses, filenameList, labelMode, plot_show)
    data_sets = DataSet(instances, labels, xlen, ylen, imagedata, step, dtype)
    return data_sets

def generateInstancesNN(instanceSize, step, numOfClasses, filenameList, 
                        mode, plot_show = 1):

    allInstances = np.array([])
    allLabels    = np.array([])
    allImages    = []
    xlen         = 0
    ylen         = 0

    image_files, bubble_num, bubble_regions = getinfo()

    for i, imageFilename in enumerate(filenameList):       
        print ('Generating instances from [' + imageFilename + 
               ']... { patch width: ' + str(instanceSize) + ' stride: ' + 
               str(step) + ' }')

        filename = gv.__DIR__ + gv.__TrainImageDir__ + imageFilename
        imageData = io.imread(filename)        
        positiveLabels = bubble_regions[image_files.index(imageFilename)]        
        [m, n, c] = imageData.shape
        Y = np.arange(0, m - m%step, step)
	X = np.arange(0, n - n%step, step)
        ylen = len(Y)
        xlen = len(X)
        
        instances, labels = patchlabel(Y,X,positiveLabels,size = instanceSize,
                                       stride = step, mode = mode)

        instances = np.append(instances, i*np.ones((instances.shape[0], 1)),
                                 axis = 1)
        allImages.append(imageData)
        # append current instance data to all instance data
        if (allInstances.size == 0):
            allInstances = instances
            allLabels    = labels
        else:
            allInstances = np.append(allInstances, instances, axis=0)
            allLabels    = np.append(allLabels, labels, axis=0)

        if(plot_show == 1):
            image_show = np.reshape(labels, (ylen, xlen))
            fig, ax = plt.subplots(2)
            ax[0].imshow(imageData)
            ax[0].set_title('Original Image')
            img = ax[1].imshow(image_show)
            ax[1].set_title('Labels')
            plt.colorbar(img)
            plt.show()

    return [allInstances, allLabels, ylen, xlen, allImages]

def patchlabel(y, x, positiveLabels, size = 40, stride = 10, mode = 'PRO',
               scale = 0.2, dtype = tf.float32):
    """
    parameters:
      Let xv, yv = numpy.meshgrid(x, y)
      (xv(i,j), yv(i,j)) represents the position of left-top corner of a small
        image patch  
      x: possibe values of X axis (array of 1*n, in type of int)
      y: possibe values of X axis (array of 1*n, in type of int)
      positiveLabels: regions of bubbles
      size: width of small image patch (as a square)
      stride: distance between two adjecent image patches
      mode: labeling method (can be labeled as sum of values of all pixes
        or values of center pixels)
      scale: adjust bandwidth of gaussian function
    """
    if(np.size(positiveLabels) == 0):
      xv, yv = np.meshgrid(x, y)
      xy = np.append(np.reshape(xv, (np.size(xv),1)), 
                     np.reshape(yv, (np.size(yv),1)), axis = 1)
      results = np.zeros((xy.shape[0],1))
      return [xy, results]
    
    MEMORY_LIMIT = 1024*1024*1024
    BYTES_PER_NUMBER = 4
    NUMBER_OF_MATRIX = 7
    
    if(mode == 'PRO'):
      xv, yv = np.meshgrid(x, y)
    elif(mode == 'NUM'):
      xv, yv = np.meshgrid(np.arange(x[-1] + stride), np.arange(y[-1] + stride))
    
    original_shape = xv.shape
    xv = np.reshape(xv,(np.size(xv),1))
    yv = np.reshape(yv,(np.size(yv),1))
    
    memory_required = (np.size(xv) * positiveLabels.shape[0] *
                       BYTES_PER_NUMBER * NUMBER_OF_MATRIX)
    num_rows = max(np.floor(MEMORY_LIMIT / BYTES_PER_NUMBER / NUMBER_OF_MATRIX
          / positiveLabels.shape[0] / original_shape[1]), 1).astype(int)
    batch_size = num_rows * original_shape[1]
    batch_num  = np.ceil(original_shape[0] / num_rows).astype(int)
     
    # place holder for tensorflow 
    _x  = tf.placeholder(dtype, shape = [batch_size, 1])
    _y  = tf.placeholder(dtype, shape = [batch_size, 1])
    _cx = tf.placeholder(dtype, shape = [1, positiveLabels.shape[0]])
    _cy = tf.placeholder(dtype, shape = [1, positiveLabels.shape[0]])
    _a  = tf.placeholder(dtype, shape = [1, positiveLabels.shape[0]])
    _b  = tf.placeholder(dtype, shape = [1, positiveLabels.shape[0]])
    
    _s  = tf.constant(stride, dtype = dtype)
    test = tf.ones([1, positiveLabels.shape[0]], dtype)
    mx  = tf.matmul(_x, tf.ones([1, positiveLabels.shape[0]], dtype))
    my  = tf.matmul(_y, tf.ones([1, positiveLabels.shape[0]], dtype))
    mcx = tf.matmul(tf.ones([batch_size, 1], dtype), _cx)
    mcy = tf.matmul(tf.ones([batch_size, 1], dtype), _cy)
    ma  = tf.matmul(tf.ones([batch_size, 1], dtype), _a)
    mb  = tf.matmul(tf.ones([batch_size, 1], dtype), _b)

    if(mode == 'PRO'):
      mx = tf.add(mx, tf.div(_s, tf.constant(2.0, dtype)))
      my = tf.add(my, tf.div(_s, tf.constant(2.0, dtype)))
    
    labels = gaussian2d(mx, my, mcx, mcy, ma, mb, tf.constant(scale), dtype)
    labels = tf.reshape(labels, [num_rows, original_shape[1]])

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    image_labels = np.zeros(original_shape)
    
    for i in range(batch_num):
      start = i * batch_size
      end   = (i+1) * batch_size
      feed  = {_x : xv[start:end], _y: yv[start:end],
              _cx: np.reshape(np.add(positiveLabels[:,0],
                      np.true_divide(positiveLabels[:,2], 2)),
                      (1, positiveLabels.shape[0])),
              _cy: np.reshape(np.add(positiveLabels[:,1],
                      np.true_divide(positiveLabels[:,3], 2)),
                      (1, positiveLabels.shape[0])),
              _a : np.reshape(positiveLabels[:,2],
                      (1, positiveLabels.shape[0])),
              _b : np.reshape(positiveLabels[:,3],
                      (1, positiveLabels.shape[0]))}
      #sess.run(labels, feed_dict=feed)
      start_row = start / original_shape[1]
      end_row   = end / original_shape[1]
      image_labels[start_row:end_row] = labels.eval(session=sess,
                                                    feed_dict=feed)

    results = image_labels
          
    if(mode == 'NUM'):
      detail   = tf.convert_to_tensor(image_labels, dtype = dtype)     
      detail   = tf.reshape(detail, [-1, image_labels.shape[1],
                            image_labels.shape[0], 1])
      template = tf.ones([size, size, 1, 1], dtype)
      sums     = tf.nn.conv2d(detail, template, [1, stride, stride, 1], "SAME")
      labels   = tf.reduce_sum(tf.reduce_sum(sums, 3), 0)
      results  = labels.eval(session=sess)
      xv, yv = np.meshgrid(x, y)

    xy = np.append(np.reshape(xv, (np.size(xv),1)), 
                   np.reshape(yv, (np.size(yv),1)), axis = 1)
    results = np.reshape(results, (np.size(results), 1))
    sess.close()
    return [xy, results]

def gaussian2d(x, y, cx, cy, a, b, scale, dtype = tf.float32):
    """
    y, x : m x n 2D tensor. Position of calculation points 
           m is number of pixels for labeling
           n is number of Gaussians in the caluclation
    cx, cy, a, b : m x n 2D tensor.
                   Parameters of Gaussian function
                   cx and cy are center position
                   a and b are the width in x and y firection
    scale: scalar tensor representing scalar factor for bandwidth of Gaussian
    """    
    A = tf.mul(tf.constant(100.0, dtype),
        tf.inv(tf.mul(tf.constant(2.0*np.pi, dtype), tf.mul(a, b))),
        name = 'Amplitude')

    powerX = tf.truediv(tf.pow(tf.sub(x, cx) , tf.constant(2.0, dtype)),
        tf.mul(tf.constant(2.0, dtype),tf.pow(a, tf.constant(2.0, dtype))))
    
    powerY = tf.truediv(tf.pow(tf.sub(y, cy) , tf.constant(2.0, dtype)),
        tf.mul(tf.constant(2.0, dtype),tf.pow(a, tf.constant(2.0, dtype))))
    
    probability = tf.reduce_sum(
        tf.mul(A, tf.exp(tf.neg(tf.add(powerX, powerY)))), 1)

    return probability

