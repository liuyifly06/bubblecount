import sys, traceback
from bubblecount.benchmark import labelaccuracy as la

def main():
    try:
        la.labellinearity(10, 1, 100, 'PRO', plot_show = 1)
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
