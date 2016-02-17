# combine adaptive and globle thereshold method according to light change
import sys, traceback, time
import numpy as np
import preprocessing as pre
import ellipse
from skimage.color import rgb2gray
from skimage import io
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData

def linear_fit_function(x, k, b):
    return k*x + b

def ord_function(beta,x):
    return beta[0]*x + beta[1]

def main():
    try:
        data_filename = 'number.txt'
        start_time = time.time()
        number = np.zeros((41,3))
        index = -1
        for det in range(1,8):
            if (det!=6):
                num_n = 6
            else:
                num_n = 5
            for n in range(0,num_n):
                index = index + 1
                temp = np.zeros(3)
                for angle in range(1,4):
                    filename = 'detector_' + str(det) + '_no_' + str(n) \
                                    + '_angle_' + str(angle) + '.jpg'
                    refname = 'detector_' + str(det) + '_no_' + str(n) \
                                    + '_background.jpg'

                    elapse_time = (time.time()-start_time)
                    if(elapse_time >= 1):
                        remain_time = elapse_time/(index*3+angle-1)*41*3-elapse_time
                        print 'Processing .. ' + filename \
                                +  time.strftime(" %H:%M:%S", time.gmtime(elapse_time)) \
                                + ' has past. ' + 'Remaining time: ' \
                                +  time.strftime(" %H:%M:%S", time.gmtime(remain_time))
                    else:
                        print 'Processing .. ' + filename \
                                +  time.strftime(" %H:%M:%S", time.gmtime(elapse_time)) \
                                + ' has past'

                    image = rgb2gray(io.imread(filename))
                    ref = rgb2gray(io.imread(refname))

                    temp[angle-1] = ellipse.count_bubble(image,ref)
                    #temp[angle-1] = ellipse.count_bubble(image)

                number[index,1] = np.mean(temp)
                number[index,2] = np.std(temp)
        
        manual_count = np.array([1,27,40,79,122,160,1,18,28,42,121,223,0,11,24,46,\
                142,173,3,19,23,76,191,197,0,15,24,45,91,152,0,\
                16,27,34,88,0,9,12,69,104,123]) 
        number[:,0] = manual_count.T
        number.tofile(data_filename,sep=" ")
  
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

if __name__ == '__main__':
    main()
