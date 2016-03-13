"""Functions for downloading and reading MNIST data."""
import tensorflow as tf
import sys, traceback, time
import numpy as np
from matplotlib import pyplot as plt
from skimage import io

class DataSet(object):

  def __init__(self, images, xlength, ylength, dtype=tf.float32):
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    self._num_examples = images.shape[0]

    if dtype == tf.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(np.float32)
      images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._xlength = xlength
    self._ylength = ylength
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

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
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end]

def read_data_sets(img, instanceSize, step, dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()
  [train_images, ylen, xlen] = generateInstances(img, instanceSize, step)
  data_sets.test = DataSet(train_images, xlen, ylen, dtype=dtype)
  return data_sets

def generateInstances(img, instanceSize, step):
	print 'Generating training instances ... instanceSize: ' \
		+ str(instanceSize) + ' step: ' + str(step)

	[m,n,c] = img.shape
	Y = np.arange(np.floor(instanceSize/2), (m - np.ceil(instanceSize/2)), step)
	X = np.arange(np.floor(instanceSize/2), (n - np.ceil(instanceSize/2)), step)
	
	totalInstancesY = len(Y)
	totalInstancesX = len(X)
	
	progressRatePrevious = 0
	Instances = np.zeros( (instanceSize ** 2 * c, totalInstancesX * totalInstancesY ))

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
        
        		temp = img[boundaryT : boundaryD, boundaryL : boundaryR, :]

        		Instances[:, ind] = np.reshape(temp,(instanceSize ** 2 * c, 1)).T
         
    		elapse_time = time.time() - start_time
   		progressRate = np.true_divide(ind, totalInstancesX*totalInstancesY) * 100
    		remain_time = np.true_divide(elapse_time, progressRate) * 100 - elapse_time
    		progressRate = np.floor(np.true_divide(progressRate, 10))
    		if (progressRate > progressRatePrevious):
        		print str(progressRate*10) + \
            		'% test instances created, remaining time: ' +\
            		time.strftime(" %H:%M:%S", time.gmtime(remain_time))
    		progressRatePrevious = progressRate

	return [Instances.T, totalInstancesY, totalInstancesX]


def main():
    try:
	test_file = '../../images/detector_2_no_5_angle_2.jpg'
	img = io.imread(test_file)
	instanceSize = 10;
	step = 2;
	read_data_sets(img, instanceSize, step)

    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()
