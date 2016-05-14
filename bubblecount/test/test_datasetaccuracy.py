import sys, traceback
from bubblecount.benchmark import datasetccuracy as la
from bubblecount.neuralnetwork import dataset as ds
def main():
    try:
        patch_size = 40
        stride =10
        labelMode = 'NUM'
        #ds.gaussianDatatoFile(patch_size, stride, labelMode)
        la.labellinearity(
                   patch_size = patch_size,
                   stride = stride,
                   labelMode = labelMode,
                   label_mutiplier = 1,
                   progress_show = 1,
                   plot_show = 1)
        """
        patch_sizes = [10, 100]
        strides = [10, 20]
        numOfClasses = [2, 10, 100]
        labelModes = ['PRO', 'NUM']
        
        labellinearity_stride(patch_sizes, strides, numOfClasses, labelModes,
                              MaxProcessNum = -1)
        """
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()
