import sys, traceback
import numpy as np
from scipy.stats import linregress
from matplotlib import pyplot as plt

def main():
    try:      
        run = True
        while(run):
            ins_size = raw_input('Input Patch Size: ')
            stride  = raw_input('Input Patch Stride: ')
            label_option = raw_input('Input Label Option: ')
            label_mode = raw_input('Input Label Mode: ')
            learning_rate = raw_input('Input Learning rate: ')
            batch_num = 10000
            batch_size = 2000
            filename = ('patch_' + str(ins_size) + '_' + str(stride)
                  + '_label_' + str(label_option) + '_' + str(label_mode)
                  + '_training_' + str(learning_rate) + '_' + str(batch_num)
                  + '_' + str(batch_size) + '_result.dat')
            y = np.fromfile(filename)
            x = np.loadtxt('manaul_count.dat')
            y.tofile('current.dat', sep = " ")
            fig, ax = plt.subplots(1,1)
            ax.scatter(x,y)
            plt.show()
            loop = raw_input('Continue ? (y/n): ')
            if(loop == 'n' or loop == 'N'):
                run = False
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()
