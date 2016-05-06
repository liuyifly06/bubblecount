# combine adaptive and globle thereshold method according to light change
import sys, traceback, time, curvature
import numpy as np
import os.path
from ..PreProcess import preprocessing as pre
from ..PreProcess.readinfo import getInfo
from .. import GlobalVariables as gv
from skimage.color import rgb2gray
from skimage import io
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData


def linear_fit_function(x, k, b):
    return k*x + b

def ord_function(beta,x):
    return beta[0]*x + beta[1]

def linearity(image_filenames, data_filename = gv.cu__result_filename):
    start_time = time.time()
    result = np.zeros((len(image_filenames), 2))
    index = -1
    
    center_region = [-400,400,-800, 600]
    
    for filename in image_filenames:
        refname = filename[0:23] + '_background.jpg'
        elapse_time = (time.time()-start_time)
        index = index + 1
          
        if(elapse_time >= 1):
            remain_time = elapse_time/index*41*3-elapse_time
            print 'Curvature analyzing .. ' + filename + \
                  time.strftime(" %H:%M:%S", time.gmtime(elapse_time))\
                  + ' has past. ' + 'Remaining time: ' + \
                  time.strftime(" %H:%M:%S", time.gmtime(remain_time))
        else:
            print 'Curvature analyzing .. ' + filename + \
                  time.strftime(" %H:%M:%S", time.gmtime(elapse_time))\
                  + ' has past'
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
 
        result[index, 0] = len(circles)
        result[index, 1] = len(circles_in_box) 

    result.tofile(data_filename, sep=" ")
    return result

def main():
    try:
        data_filename = gv.cu__result_filename
        image_files, bubble_num, bubble_regions = getInfo()
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
