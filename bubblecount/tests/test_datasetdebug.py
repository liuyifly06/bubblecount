import sys, traceback, time, warnings
from bubblecount.neuralnetwork import dataset as ds
from bubblecount.preprocess.readinfo import getinfo
import numpy as np

def main():
     try:
         instanceSize = 10
         step = 1
	 numOfClasses = 100
         filename = 'detector_1_no_5_angle_3.jpg'
         mode = 'NUM'
         image_files, bubble_num, bubble_regions = getinfo()
         data_set = ds.read_data_sets(instanceSize, step, numOfClasses,
           'test', mode, filename, plot_show = 1)
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
