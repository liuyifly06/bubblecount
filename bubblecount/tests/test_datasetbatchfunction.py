import sys, traceback, time, warnings
from bubblecount.neuralnetwork import dataset as ds

"""
test if dataset.py functioning well  
"""

def main():
     try:
         instanceSize = 20
         step = 20
	 numOfClasses = 100
         
         print 'Testing data generation ... '
         data_set = ds.read_data_sets(instanceSize, step, numOfClasses,
           'train', 'PRO', plot_show = 0)
         print 'Instances shape :'
         print data_set.images.shape
         print 'label shape :'
         print data_set.labels.shape
         print 'Testing batch data generation :'
        
         batch_size = 1000
         batch_num  = 1000
         for i in range(batch_num):
            print ('batch num: ' + str(i) +
                  str(data_set.next_batch(batch_size)[0].shape) +
                  str(data_set.next_batch(batch_size)[1].shape))
         ds.read_data_sets(instanceSize, step, numOfClasses,
           'test', 'PRO', 'detector_1_no_5_angle_3.jpg', plot_show = 1)
         ds.read_data_sets(instanceSize, step, numOfClasses, 'train', 'NUM',
           plot_show = 1)
         ds.read_data_sets(instanceSize, step, numOfClasses,
           'test', 'NUM', 'detector_1_no_5_angle_3.jpg', plot_show = 1)

         
     except KeyboardInterrupt:
         print "Shutdown requested... exiting"
     except Exception:
         traceback.print_exc(file=sys.stdout)
     sys.exit(0)
 
if __name__ == '__main__':
     main()
