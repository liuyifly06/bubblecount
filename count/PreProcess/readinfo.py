import sys, traceback, time
import numpy as np
from matplotlib import pyplot as plt
from .. import GlobalVariables as gv

def getInfo(filename = gv.__DIR__ + gv.__TrainImageDir__ + \
            'positiveInstances.dat'):
    f_read = open(filename, 'r')
    image_files    = []
    bubble_num     = np.array([])
    bubble_regions = []
    for line in f_read:
        line_list = line.split()
        image_files.append(line_list[0])
        bubble_num = np.append(bubble_num, int(line_list[1]))
        
        bubble_region = np.reshape(map(int, line_list[2:]), \
                                       (int(line_list[1]), 4) )
        bubble_regions.append(bubble_region)
    f_read.close()
    return [image_files, bubble_num, bubble_regions]

def distribution(binwidth = 5):
    image_files, bubble_num, bubble_regions = getInfo()
    
    bubble_all = np.array([])
    for bubble_region in bubble_regions:
        bubble_all = np.append(bubble_all, np.true_divide(\
            np.add(bubble_region[:,2], bubble_region[:,3]), 2))

    fig, ax = plt.subplots(1,2)
    ax[0].hist(bubble_num, bins=np.arange(min(bubble_num), \
                    max(bubble_num) + binwidth, binwidth))
    ax[0].set_title("Bubble Number Histogram")
    ax[0].set_xlabel("Number of bubbles in each image")
    ax[0].set_ylabel("Number of images")
    ax[1].hist(bubble_all, bins=np.arange(min(bubble_all), \
                    max(bubble_all) + binwidth, binwidth))
    ax[1].set_title("Bubble Diameter Histogram")
    ax[1].set_xlabel("Bubble diameter")
    ax[1].set_ylabel("Number of bubbles")
    plt.show()  

def main():
    try:
        image_files, bubble_num, bubble_regions = getInfo()
        f = open('manaul_count.dat','w')
        bubble_num.tofile(f, sep = ' ')
        f.close()
        distribution()
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()
