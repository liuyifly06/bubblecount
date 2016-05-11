import sys, traceback, time, warnings
from bubblecount.neuralnetwork import dataset as ds
from bubblecount.preprocess.readinfo import getinfo
import numpy as np
import tensorflow as tf
def main():
     try:
         instanceSize = 10
         step = 1
	 numOfClasses = 100
         filename = 'detector_7_no_5_angle_2.jpg'
         mode = 'PRO'
         image_files, bubble_num, bubble_regions = getinfo()
         data_set = ds.read_data_sets(
                   instanceSize = instanceSize,
                   stride = step,
                   instanceMode = 'test', 
                   labelMode = mode,
                   imageName = filename,
                   dtype = tf.float32,
                   plot_show = 0,
                   label_mutiplier = 1.0)
         print data_set.labels.shape[-1]
         print 'Sum of labels:  ' + str(np.sum(data_set.labels))
         print 'Number bubbles: ' + str(bubble_num[image_files.index(filename)])

     except KeyboardInterrupt:
         print "Shutdown requested... exiting"
     except Exception:
         traceback.print_exc(file=sys.stdout)
     sys.exit(0)
 
if __name__ == '__main__':
     main()
