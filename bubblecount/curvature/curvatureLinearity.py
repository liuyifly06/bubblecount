# combine adaptive and globle thereshold method according to light change
import sys, traceback, time, curvature
import numpy as np
import os.path
from skimage.color import rgb2gray
from skimage import io
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData
import bubblecount.globalvar as gv
from bubblecount.preprocess import preprocessing as pre
from bubblecount.preprocess.readinfo import getinfo
from bubblecount.progressbar import progress

def linearity(image_filenames, data_filename = gv.cu__result_filename):
    result = np.zeros((len(image_filenames), 2))
    center_region = [-400,400,-800,600]
    probar = progress.progress(0, len(image_filenames))
    for iteration, filename in enumerate(image_filenames):
        probar.setCurrentIteration(iteration+1)
        probar.setInfo(
            prefix_info = 'Curvature linearity ...',
            suffix_info = filename)
        probar.printProgress()
        refname = filename[0:23] + '_background.jpg'
        image = io.imread(gv.__DIR__ + gv.__TrainImageDir__ + filename)
        OY = int(image.shape[0]/2)
        OX = int(image.shape[1]/2)

        margin = [max(OY + center_region[0], 0), \
                  min(OY + center_region[1], image.shape[0]), \
                  max(OX + center_region[2], 0), \
                  min(OX + center_region[3], image.shape[1])]
        
        circles = curvature.count_bubble(filename, refname, plot_show = 0)
        index_in_box = np.all(
          [np.all([circles[:, 0] <= margin[1], circles[:, 0] >= margin[0]], \
                  axis = 0), \
           np.all([circles[:, 1] <= margin[3], circles[:, 1] >= margin[2]], \
                  axis = 0)],  axis = 0)
        circles_in_box = circles[index_in_box]
 
        result[iteration, 0] = len(circles)
        result[iteration, 1] = len(circles_in_box) 

    result.tofile(data_filename, sep=" ")
    return result

def main():
    try:
        data_filename = gv.cu__result_filename
        image_files, bubble_num, bubble_regions = getinfo()
        if not os.path.isfile(data_filename):
            number = linearity(image_files)
        number = np.loadtxt(data_filename)
        number = np.reshape(number, (len(number)/2,2))

        fig, ax = plt.subplots(1,1)
        ax.scatter(bubble_num, number[:, 0], c = 'r')
        k, b = np.polyfit(bubble_num, number[:, 0], 1)
        ax.plot(bubble_num, bubble_num*k+b, 'r', linewidth = 2)
        ax.scatter(bubble_num, number[:, 1], c = 'b')
        k, b = np.polyfit(bubble_num, number[:, 1], 1)
        ax.plot(bubble_num, bubble_num*k+b, 'b', linewidth = 2)
        plt.show()
  
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

if __name__ == '__main__':
    main()
