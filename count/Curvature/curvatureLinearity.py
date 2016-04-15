# combine adaptive and globle thereshold method according to light change
import sys, traceback, time
import numpy as np
import os.path
from ..PreProcess import preprocessing as pre
from ..PreProcess.readinfo import getInfo
import curvature
from skimage.color import rgb2gray
from skimage import io
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData


def linear_fit_function(x, k, b):
    return k*x + b

def ord_function(beta,x):
    return beta[0]*x + beta[1]

def linearity(length = 123, data_filename = 'curvature.txt'):
    directory = '/home/yi/Documents/bubble-counting/images/AfterPreprocessing/'
    start_time = time.time()
    number = np.zeros((length,1))
    index = -1
    for det in range(1,8):
        if (det!=6):
            num_n = 6
        else:
            num_n = 5
        for n in range(0,num_n):
            for angle in range(1,4):
                index = index + 1

                filename = 'detector_' + str(det) + '_no_' + str(n) \
                             + '_angle_' + str(angle) + '.jpg'
                refname = 'detector_' + str(det) + '_no_' + str(n) \
                             + '_background.jpg'

                elapse_time = (time.time()-start_time)
                if(elapse_time >= 1):
                    remain_time = elapse_time/index*41*3-elapse_time
                    print 'Curvature analyzing .. ' + filename \
                           +  time.strftime(" %H:%M:%S", time.gmtime(elapse_time)) \
                           + ' has past. ' + 'Remaining time: ' \
                           +  time.strftime(" %H:%M:%S", time.gmtime(remain_time))
                else:
                    print 'Curvature analyzing .. ' + filename \
                           +  time.strftime(" %H:%M:%S", time.gmtime(elapse_time)) \
                           + ' has past'

                image = rgb2gray(io.imread(directory+filename))
                #ref = rgb2gray(io.imread(directory+refname))

                number[index] = curvature.count_bubble(image)
                #temp[angle-1] = ellipse.count_bubble(image)
    number.tofile(data_filename,sep=" ")
    return number

def main():
    try:
        data_filename = 'curvature.txt'
        image_files, bubble_num, bubble_regions = getInfo()
        if not os.path.isfile(data_filename):
            number = linearity(len(image_files))
        number = np.loadtxt(data_filename)
        k, b = np.polyfit(bubble_num, number, 1)
  
        fig, ax = plt.subplots(1,1)
        ax.scatter(bubble_num, number)
        ax.plot(bubble_num, bubble_num*k+b, 'r', linewidth = 2)
        plt.show()
  
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

if __name__ == '__main__':
    main()
