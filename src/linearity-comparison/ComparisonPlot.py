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
        data = np.loadtxt(data_filename,skiprows=0)
        cur = np.reshape(data,(41,3))

        data_filename = 'curvature.txt'
        data = np.loadtxt(data_filename,skiprows=0)
        hough = np.reshape(data,(41,3))
    
        #emperical error    
        xerr = np.sqrt(hough[:,0])/3
        yerr_h = hough[:,2]
	yerr_c = cur[:,2]

        data_h = Data(hough[:,0].T, hough[:,1].T, we = 1/(np.power(xerr.T,2)+np.spacing(1)), wd = 1/(np.power(yerr_h.T,2)+np.spacing(1)))
        data_c = Data(cur[:,0].T, cur[:,1].T, we = 1/(np.power(xerr.T,2)+np.spacing(1)), wd = 1/(np.power(yerr_c.T,2)+np.spacing(1)))
	
        model = Model(ord_function)
        odr_h = ODR(data_h, model, beta0=[0, 0])
	odr_c = ODR(data_c, model, beta0=[0, 0])

        odr_h.set_job(fit_type=2)
	odr_c.set_job(fit_type=2)

        output_h = odr_h.run()
	output_c = odr_c.run()
    
        popt_h = output_h.beta
        perr_h = output_h.sd_beta

        popt_c = output_c.beta
        perr_c = output_c.sd_beta
        
 	popt_h, pcov_h = curve_fit(linear_fit_function, hough[:,0], hough[:,1], [1, 0], hough[:, 2])
        perr_h = np.sqrt(np.diag(pcov_h))

# 	popt_c, pcov_c = curve_fit(linear_fit_function, cur[:,0], cur[:,1], [1, 0], cur[:, 2])
#       perr_c = np.sqrt(np.diag(pcov_c))
	
	A = popt_h[0]/np.sqrt(popt_h[0]*popt_h[0]+1)
	B = -1/np.sqrt(popt_h[0]*popt_h[0]+1)
	C = popt_h[1]/np.sqrt(popt_h[0]*popt_h[0]+1)
        fitting_error_h = np.mean(np.abs(A*hough[:,0]+B*hough[:,1]+C))
	
	A = popt_c[0]/np.sqrt(popt_c[0]*popt_c[0]+1)
	B = -1/np.sqrt(popt_c[0]*popt_c[0]+1)
	C = popt_c[1]/np.sqrt(popt_c[0]*popt_c[0]+1)
	fitting_error_c = np.mean(np.abs(A*cur[:,0]+B*cur[:,1]+C))
    
        fig, ax = plt.subplots(ncols = 1)
        ax.errorbar(hough[:,0], hough[:,1], xerr = xerr, yerr = yerr_h, fmt='o',color='blue')
	ax.errorbar(cur[:,0], cur[:,1], xerr = xerr, yerr = yerr_c, fmt='o',color='red')
        
	ax.plot(hough[:,0], popt_h[0]*hough[:,0]+popt_h[1], '-b',linewidth = 2)
	ax.plot(cur[:,0], popt_c[0]*cur[:,0]+popt_c[1], '-r',linewidth = 2)
    
        bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=2)

        annotation_text = "function:  y = kx + b \n" \
        "Hough Transfrom (blue)\n"\
	"k = %.2f b = %.2f Error = %.2f" % (popt_h[0], popt_h[1], fitting_error_h) + '\n'\
        "Curvature Method (red)\n"\
	"k = %.2f b = %.2f Error = %.2f" % (popt_c[0], popt_c[1], fitting_error_c)
        
	ax.text(10, max(np.amax(hough[:,1]), np.amax(cur[:,1]))+10, annotation_text, ha="left", va="top", rotation=0,
    size=15, bbox=bbox_props)
       
        ax.set_title('Algorithom Performance')
        ax.set_xlabel('Bubble Number Counted Manually')
        ax.set_ylabel('Bubbble Number Counted by Algorithom')
        plt.grid()
        plt.xlim((np.amin(hough[:,0])-5,np.amax(hough[:,0])+5))
        plt.ylim((0,max(np.amax(hough[:,1]), np.amax(cur[:,1]))+20))
        plt.show()
  
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

if __name__ == '__main__':
    main()
