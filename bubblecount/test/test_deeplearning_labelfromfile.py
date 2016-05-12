import sys, traceback
from bubblecount.neuralnetwork import dataset as ds
def main():
    try:
        ds.generateInstancesNN(instanceSize = 40,
                        stride = 10,
                        filenameList = ['detector_5_no_5_angle_2.jpg'], 
                        mode = 'PRO', plot_show = 1,
                        label_mutiplier = 1.0)
        
        #"""
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()
