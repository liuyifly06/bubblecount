"""Functions for downloading and reading MNIST data."""
import tensorflow as tf
import sys, traceback, time
import numpy as np
from matplotlib import pyplot as plt
from skimage import io

class DataSet(object):

  def __init__(self, images, labels, dtype=tf.float32):
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

def read_data_sets(instanceSize, step, dtype=tf.float32, plot_show = 0):
  class DataSets(object):
    pass
  data_sets = DataSets()
  [train_images, train_labels] = generateInstancesNN(instanceSize, step, plot_show)
  data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
  return data_sets

def generateInstancesNN(instanceSize, step, plot_show = 1):
	print 'Generating training instances ... instanceSize: ' \
		+ str(instanceSize) + ' step: ' + str(step)

	train_file = "../../images/detector_5_no_5_angle_2.jpg"
	train = io.imread(train_file)
	positiveLabels = poistiveLabelRegion()
	[m,n,c] = train.shape

	Y = np.arange(np.floor(instanceSize/2), (m - np.ceil(instanceSize/2)), step)
	X = np.arange(np.floor(instanceSize/2), (n - np.ceil(instanceSize/2)), step)
	
	totalInstancesY = len(Y)
	totalInstancesX = len(X)
	
	progressRatePrevious = 0

	Instances = np.zeros( (instanceSize ** 2 * c, totalInstancesX * totalInstancesY ))
	Labels = np.zeros((100, totalInstancesX * totalInstancesY))

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

        		Instances[:, ind] = np.reshape(temp,(instanceSize ** 2 * c, 1)).T
			
			probabiliyIndex = max(0,np.floor(np.sqrt(\
				gaussian2d(currentY, currentX, positiveLabels, \
				scale = 0.3))*100)-1);
			probabiliyIndex = min(99, probabiliyIndex);
			Labels[probabiliyIndex, ind] = 1;
    
    		elapse_time = time.time() - start_time
   		progressRate = np.true_divide(ind, totalInstancesX*totalInstancesY) * 100
    		remain_time = np.true_divide(elapse_time, progressRate) * 100 - elapse_time
    		progressRate = np.floor(np.true_divide(progressRate, 10))
    		if (progressRate > progressRatePrevious):
        		print str(progressRate*10) + \
            		'% train instances created, remaining time: ' +\
            		time.strftime(" %H:%M:%S", time.gmtime(remain_time))
    		progressRatePrevious = progressRate

	if(plot_show == 1):
		fig, ax = plt.subplots(2)
		ax[0].imshow(train)
		ax[0].set_title('Training Image')
		ax[1].imshow(np.reshape(np.sum( \
			np.multiply(Labels.T,np.arange(100)).T,0),\
			(totalInstancesY,totalInstancesX)))
		ax[1].set_title('Labels')
		plt.show()
	return [Instances.T, Labels.T]

def gaussian2d(y, x, positiveLabels, scale = 0.2):

	x_c = positiveLabels[:,0] + np.ceil(np.true_divide(positiveLabels[:,2],2))
	y_c = positiveLabels[:,1] + np.ceil(np.true_divide(positiveLabels[:,3],2))
	a = np.ceil(np.true_divide(positiveLabels[:,2],2)) * scale
	b = np.ceil(np.true_divide(positiveLabels[:,3],2)) * scale

	A = 100*np.true_divide(1, 2*np.pi*a*b)
#	A = 1
	powerX = np.true_divide(np.power(np.subtract(x,x_c),2) \
			, np.multiply(2, np.power(a,2))) 
	powerY = np.true_divide(np.power(np.subtract(y,y_c),2) \
			, np.multiply(2, np.power(b,2))) 
	
	Pr = np.multiply(A, np.exp(-(powerX+powerY)))
	
	return np.sum(Pr) 

def poistiveLabelRegion():
	return np.array([[1565, 1441, 59, 61], \
	[1311, 788, 44, 46], \
	[2113, 1035, 13, 14], \
	[1955, 778, 27, 24], \
	[1948, 990, 72, 73], \
	[1986, 1040, 30, 31], \
	[3110, 1444, 36, 38], \
	[3508, 1124, 40, 39], \
	[3195, 992, 27, 31], \
	[3201, 968, 34, 35], \
	[3238, 985, 32, 31], \
	[3164, 987, 25, 33], \
	[3417, 970, 31, 32], \
	[3361, 975, 35, 38], \
	[3339, 1343, 27, 34], \
	[3274, 1237, 26, 31], \
	[3270, 1171, 33, 35], \
	[3246, 1118, 25, 30], \
	[3222, 1090, 27, 32], \
	[3147, 1131, 26, 32], \
	[3029, 1143, 33, 35], \
	[3012, 1143, 30, 37], \
	[2988, 1082, 30, 35], \
	[2962, 1167, 26, 37], \
	[2928, 1171, 29, 34], \
	[2881, 1272, 28, 33], \
	[2791, 1356, 29, 31], \
	[2929, 1037, 29, 31], \
	[2841, 1090, 35, 34], \
	[3026, 926, 32, 36], \
	[2959, 948, 31, 37], \
	[2754, 1041, 39, 42], \
	[2798, 956, 27, 34], \
	[2683, 987, 39, 42], \
	[2742, 894, 31, 32], \
	[2719, 907, 23, 25], \
	[2667, 872, 30, 34], \
	[2660, 866, 27, 28], \
	[2650, 851, 36, 34], \
	[2615, 916, 30, 31], \
	[2574, 952, 31, 33], \
	[2551, 921, 31, 28], \
	[2525, 903, 31, 32], \
	[2630, 807, 25, 27], \
	[2637, 797, 26, 30], \
	[2697, 781, 29, 31], \
	[2927, 844, 27, 28], \
	[2868, 806, 28, 32], \
	[3050, 856, 35, 38], \
	[2985, 798, 29, 34], \
	[2936, 790, 29, 30], \
	[2779, 757, 28, 31], \
	[2851, 690, 25, 33], \
	[2833, 643, 30, 38], \
	[2838, 597, 26, 33], \
	[3162, 863, 32, 32], \
	[3161, 816, 33, 34], \
	[3158, 739, 32, 35], \
	[3289, 692, 32, 38], \
	[3273, 619, 32, 34], \
	[3156, 701, 30, 33], \
	[2966, 673, 30, 31], \
	[3034, 661, 28, 30], \
	[3049, 559, 36, 47], \
	[2709, 675, 28, 31], \
	[2728, 620, 29, 29], \
	[2751, 602, 26, 28], \
	[2675, 533, 35, 35], \
	[2602, 656, 39, 44], \
	[2485, 807, 34, 33], \
	[2491, 749, 32, 36], \
	[2471, 641, 31, 28], \
	[2192, 619, 31, 33], \
	[2657, 1164, 26, 30], \
	[2602, 1034, 30, 36], \
	[2586, 1040, 31, 38], \
	[2512, 1067, 32, 35], \
	[2481, 1115, 57, 56], \
	[2549, 1182, 32, 33], \
	[2402, 1179, 28, 29], \
	[2424, 1031, 34, 32], \
	[2377, 1017, 30, 30], \
	[2339, 1096, 34, 37], \
	[2278, 1109, 34, 34], \
	[2302, 1048, 33, 32], \
	[2255, 1086, 30, 28], \
	[2274, 1055, 33, 37], \
	[2332, 969, 28, 31], \
	[2317, 967, 28, 35], \
	[2404, 824, 23, 24], \
	[2317, 865, 29, 32], \
	[2301, 877, 31, 31], \
	[2333, 804, 35, 38], \
	[2293, 794, 31, 31], \
	[2275, 812, 32, 32], \
	[2246, 895, 30, 28], \
	[2150, 856, 77, 80], \
	[2136, 1015, 27, 30], \
	[2069, 1006, 91, 86], \
	[2106, 966, 31, 34], \
	[2095, 977, 33, 33], \
	[1664, 688, 55, 54], \
	[1654, 770, 57, 63], \
	[2482, 1209, 39, 40], \
	[2495, 1265, 62, 63], \
	[2317, 1380, 63, 64], \
	[2291, 1440, 24, 29], \
	[2285, 1380, 27, 27], \
	[2225, 1222, 70, 79], \
	[2230, 1184, 27, 32], \
	[2166, 1138, 20, 23], \
	[2059, 1120, 33, 36], \
	[2162, 1246, 28, 32], \
	[2032, 1250, 90, 90], \
	[2001, 1208, 28, 31], \
	[1959, 1304, 32, 35], \
	[1884, 1336, 30, 30], \
	[1695, 1354, 49, 53], \
	[1673, 1361, 33, 35], \
	[1728, 1304, 32, 34], \
	[1775, 1200, 36, 36], \
	[2043, 990, 27, 30], \
	[1884, 972, 24, 22], \
	[1931, 992, 23, 26], \
	[1887, 1008, 36, 32], \
	[1908, 1032, 31, 34], \
	[1928, 1140, 33, 34], \
	[1805, 1029, 34, 37], \
	[1801, 1113, 24, 27], \
	[1786, 1102, 26, 26], \
	[1732, 1076, 22, 21], \
	[1730, 1120, 70, 68], \
	[1705, 1136, 34, 34], \
	[1621, 914, 104, 107], \
	[1804, 919, 32, 31], \
	[2026, 923, 24, 24], \
	[2004, 855, 70, 72], \
	[2177, 836, 23, 22], \
	[2127, 801, 36, 43], \
	[2236, 722, 30, 31], \
	[2167, 761, 28, 30], \
	[2090, 733, 32, 36], \
	[2011, 766, 31, 34], \
	[1972, 759, 25, 28], \
	[2159, 649, 27, 32], \
	[2074, 664, 28, 27], \
	[1978, 626, 88, 84], \
	[1815, 698, 26, 27], \
	[1910, 562, 32, 37], \
	[1650, 457, 51, 60]])
def main():
     try:
         instanceSize = 10;
         step = 2;
         edge = 4;
         scale =10;
	 read_data_sets(instanceSize, step, plot_show = 1)
 
     except KeyboardInterrupt:
         print "Shutdown requested... exiting"
     except Exception:
         traceback.print_exc(file=sys.stdout)
     sys.exit(0)
 
if __name__ == '__main__':
     main()

