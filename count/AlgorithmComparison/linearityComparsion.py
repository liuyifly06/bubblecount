# compare linearity of different method 
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

def show():
    data_filename = 'number.txt'
    data = np.loadtxt(data_filename,skiprows=0)
    number = np.reshape(data,(41,3))
    
    #emperical error    
    xerr = np.sqrt(number[:,0])/3
    yerr = number[:,2]
    data = Data(number[:,0].T, number[:,1].T, we = 1/(np.power(xerr.T,2)+np.spacing(1)), wd = 1/(np.power(yerr.T,2)+np.spacing(1)))
    
    model = Model(ord_function)
    odr = ODR(data, model, beta0=[0, 0])
    odr.set_job(fit_type=2)
    output = odr.run()
    
    popt = output.beta
    perr = output.sd_beta

    output.pprint()

    fitting_error = np.mean(np.sqrt(np.power(popt[0]*number[:,0]+popt[1] - number[:,1],2)))
    
    
    labels = np.array([[1,1,1,1,1,1,\
           2,2,2,2,2,2,\
           3,3,3,3,3,3,\
           4,4,4,4,4,4,\
           5,5,5,5,5,5,\
           6,6,6,6,6,\
           7,7,7,7,7,7],
          [0,1,2,3,4,5,\
           0,1,2,3,4,5,\
           0,1,2,3,4,5,\
           0,1,2,3,4,5,\
           0,1,2,3,4,5,\
           0,1,2,3,4,
           0,1,2,3,4,5]])
    
    fig, ax = plt.subplots(ncols = 1)
    ax.errorbar(number[:,0], number[:,1], xerr = xerr, yerr = yerr, fmt='o')

    ax.plot(number[:,0], popt[0]*number[:,0]+popt[1], '-r')
    
    bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=2)

    annotation_text = "function:  y = kx + b \n" \
        "k = %.2f" % popt[0] + " +/- %.2f" % perr[0] + '\n' \
        "b = %.2f" % popt[1] + " +/- %.2f" % perr[1] + '\n' \
        "Error: %.2f" % fitting_error
    ax.text(10, np.amax(number[:,1])+10, annotation_text, ha="left", va="top", rotation=0,
    size=15, bbox=bbox_props)
    
    for i in range(0, len(number[:,0])):            # <--
        ax.annotate('(%s, %s)' % (labels[0,i], labels[1,i]),\
        (number[i,0]+1, number[i,1]-1)) #
    
    
    ax.set_title('Algorithom Performance')
    ax.set_xlabel('Bubble Number Counted Manually')
    ax.set_ylabel('Bubbble Number Counted by Algorithom')
    plt.grid()
    plt.xlim((np.amin(number[:,0])-5,np.amax(number[:,0])+5))
    plt.ylim((0,np.amax(number[:,1])+20))
    plt.show()
    
def main():
    try:
        if (len(sys.argv) > 1 and sys.argv[1] == '1'):
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
        elif(len(sys.argv) > 1):
            data_filename = sys.argv[1]
            data = np.loadtxt(data_filename,skiprows=0)
            number = np.reshape(data,(41,3))
        else:
            data_filename = 'number.txt'
            data = np.loadtxt(data_filename,skiprows=0)
            number = np.reshape(data,(41,3))
    
        #emperical error    
        xerr = np.sqrt(number[:,0])/3
        yerr = number[:,2]
        data = Data(number[:,0].T, number[:,1].T, we = 1/(np.power(xerr.T,2)+np.spacing(1)), wd = 1/(np.power(yerr.T,2)+np.spacing(1)))
    
        model = Model(ord_function)
        odr = ODR(data, model, beta0=[0, 0])
        odr.set_job(fit_type=2)
        output = odr.run()
    
        popt = output.beta
        perr = output.sd_beta

        output.pprint()
    
#       popt, pcov = curve_fit(linear_fit_function, number[:,0], number[:,1], [0, 0], number[:, 2])
#       perr = np.sqrt(np.diag(pcov))
        fitting_error = np.mean(np.sqrt(np.power(popt[0]*number[:,0]+popt[1] - number[:,1],2)))
    
    
        labels = np.array([[1,1,1,1,1,1,\
           2,2,2,2,2,2,\
           3,3,3,3,3,3,\
           4,4,4,4,4,4,\
           5,5,5,5,5,5,\
           6,6,6,6,6,\
           7,7,7,7,7,7],
          [0,1,2,3,4,5,\
           0,1,2,3,4,5,\
           0,1,2,3,4,5,\
           0,1,2,3,4,5,\
           0,1,2,3,4,5,\
           0,1,2,3,4,
           0,1,2,3,4,5]])
    
        fig, ax = plt.subplots(ncols = 1)
        ax.errorbar(number[:,0], number[:,1], xerr = xerr, yerr = yerr, fmt='o')

        ax.plot(number[:,0], popt[0]*number[:,0]+popt[1], '-r')
    
        bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=2)

        annotation_text = "function:  y = kx + b \n" \
        "k = %.2f" % popt[0] + " +/- %.2f" % perr[0] + '\n' \
        "b = %.2f" % popt[1] + " +/- %.2f" % perr[1] + '\n' \
        "Error: %.2f" % fitting_error
        ax.text(10, np.amax(number[:,1])+10, annotation_text, ha="left", va="top", rotation=0,
    size=15, bbox=bbox_props)
    
        for i in range(0, len(number[:,0])):            # <--
            ax.annotate('(%s, %s)' % (labels[0,i], labels[1,i]),\
                (number[i,0]+1, number[i,1]-1)) #
    
    
        ax.set_title('Algorithom Performance')
        ax.set_xlabel('Bubble Number Counted Manually')
        ax.set_ylabel('Bubbble Number Counted by Algorithom')
        plt.grid()
        plt.xlim((np.amin(number[:,0])-5,np.amax(number[:,0])+5))
        plt.ylim((0,np.amax(number[:,1])+20))
        plt.show()
  
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

if __name__ == '__main__':
    main()
