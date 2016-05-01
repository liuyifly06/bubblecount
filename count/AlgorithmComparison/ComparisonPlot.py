# combine adaptive and globle thereshold method according to light change
import sys, traceback, time
import numpy as np
from skimage.color import rgb2gray
from skimage import io
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData
from .. import GlobalVariables as gv
from scipy.stats import linregress
def linear_fit_function(x, k, b):
    return k*x + b

def ord_function(beta,x):
    return beta[0]*x + beta[1]

def manual_count():
    filename = gv.__DIR__ + gv.pp__image_dir+'positiveInstances.dat'
    f_read = open(filename, 'r')
    index = 0
    data = []
    for line in f_read:
       line_list = line.split()
       data.append(int(line_list[1]))
    f_read.close()
    return data

def xaxis(data):
    res = np.zeros((len(data)/3, 2))
    for i in range(len(data)/3):
       res[i][0] = np.mean(data[i*3 : (i*3 + 2)])
       res[i][1] = np.std(data[i*3 : (i*3 + 2)])
    return res

def main():
    try:
        manual_number = manual_count()
        xvalue = xaxis(manual_number)
        legend = ['curvature',
                  'neural-network',
                  'hough transform',
                  'SVM']
	data_filename = ['number.txt',
                         'skflow.txt',
                         'curvature.txt',
                         'data.txt']
        colors = ['blue', 'red', 'black', 'green']

        annotation_text = "function:  y = kx + b";
        fig, ax = plt.subplots(ncols = 1)
        for i in range(len(data_filename)):
            temp = np.loadtxt(gv.__DIR__ + gv.ac__file_dir  + data_filename[i],
                              skiprows=0)
            data = np.reshape(temp,(41,len(temp)/41))
            #print data
            data[:,1:3] = np.true_divide(data[:,1:3], np.amax(data[:,1]))
            data[:,1] = data[:,1]+i
            #print data
            #emperical error 
            xerr = np.sqrt(data[:,0])/3
            yerr = data[:,2]
            data_fit = Data(data[:,0].T, data[:,1].T, \
                          we = 1/(np.power(xerr.T,2)+np.spacing(1)),\
                          wd = 1/(np.power(yerr.T,2)+np.spacing(1)))

            model = Model(ord_function)
            odr = ODR(data_fit, model, beta0=[0, 0])
	    odr.set_job(fit_type=2)
	    output = odr.run()
            popt = output.beta
            perr= output.sd_beta
            popt, pcov = curve_fit(linear_fit_function, 
                             data[:,0], data[:,1], [1, 0], data[:, 2])
            perr = np.sqrt(np.diag(pcov))

	    A = popt[0]/np.sqrt(popt[0]*popt[0]+1)
	    B = -1/np.sqrt(popt[0]*popt[0]+1)
	    C = popt[1]/np.sqrt(popt[0]*popt[0]+1)
            fitting_error= np.mean(np.abs(A*data[:,0]+B*data[:,1]+C))

            ax.errorbar(data[:,0], data[:,1], xerr = xerr,\
                        yerr = yerr, fmt='o',color=colors[i])
            ### not using error bar in fitting #########
            slope, intercept, r_value, p_value, std_err = (
                   linregress(xvalue[:,0], data[:,1]))
              
            #popt[0],popt[1] = np.polyfit(xvalue[:,0], data[:,1], 1)
            ###
            ax.plot(data[:,0], slope*data[:,0]+intercept,
                    colors[i], linewidth = 2)
            annotation_text = annotation_text + "\n" +\
                              legend[i] + "(" + \
                              colors[i] + ") \n" + \
                              "k = %.2f b = %.2f Error = %.2f"%( \
                              popt[0], popt[1], fitting_error) 
            print[slope, intercept, r_value, p_value, std_err], data_filename[i]
        bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=2)
	#ax.text(0, 4.15, annotation_text, ha="left", va="top", \
        #        rotation=0, size=14, bbox=bbox_props)
   
        ax.set_title('Algorithom Performance')
        ax.set_xlabel('Bubble Number Counted Manually')
        ax.set_ylabel('Normalized Bubbble Number Counted by Algorithom')
        plt.grid()
        plt.xlim((np.amin(data[:,0])-5,np.amax(data[:,0])+5))
        plt.ylim((0,np.amax(data[:,1])+0.2))
        plt.show()
  
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

if __name__ == '__main__':
    main()
