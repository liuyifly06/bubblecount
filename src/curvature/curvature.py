# Note: Combination of connection points which are too close
# Note: Ellipse Parameter Recognition
# Note: Eliminate Noninterested Part

import sys, traceback, time
import numpy as np
import preprocessing as pre

from scipy import ndimage as ndi
from scipy import signal, interpolate
from scipy.fftpack import rfft, irfft, fftfreq
from matplotlib import pyplot as plt
from math import sqrt, sin, exp, atan, atan2

from skimage import io, filters, measure
from skimage.color import rgb2gray
from skimage.feature import canny, peak_local_max, match_template
from skimage.transform import hough_ellipse, hough_circle, rotate
from skimage.draw import ellipse_perimeter, circle_perimeter
from skimage.util import img_as_ubyte, img_as_float
from skimage.morphology import watershed, disk ,remove_small_objects
from skimage.segmentation import random_walker

def parameter_estimation(seg, der, mode = 0):
    # mode = 0 for circle; 1 for ellipse
    if(mode == 0):
        A = np.add(np.multiply(der[:,0],seg[:,0]),np.multiply(der[:,1],seg[:,1]))
        XY = np.sum(np.multiply(der[:,0],der[:,1]))
        X2 = np.sum(np.multiply(der[:,0],der[:,0]))
        Y2 = np.sum(np.multiply(der[:,1],der[:,1]))
        XA = np.sum(np.multiply(der[:,0],A))
        YA = np.sum(np.multiply(der[:,1],A))

        x_c = (XA*Y2 - XY*YA) / (X2*Y2 - XY*XY)
        y_c = (YA*X2 - XY*XA) / (X2*Y2 - XY*XY)

        r = np.mean(np.sqrt(np.square(np.subtract(seg[:,0],x_c))+np.square(np.subtract(seg[:,1],y_c))))
        distance =  np.mean(np.square(np.sqrt(np.square(seg[:,0]-x_c)+np.square(seg[:,1]-y_c))-r))
        return [x_c, y_c, r, distance]
    

def iff_filter(sig, scale, plot_show = 0):
    
    order = max(sig.size*scale,90)
    #order = 80
    # Extend signal on both sides for removing boundary effect in convolution
    sig_extend = np.ones(sig.size+int(order/2)*2)
    sig_extend[int(order/2):(sig.size+int(order/2))] = sig
    sig_extend[0:int(order/2)] = sig[(sig.size-int(order/2)):sig.size]
    sig_extend[(sig.size+int(order/2)):sig_extend.size] = sig[0:int(order/2)]
    
    # convolve with hamming window and normalize
    smooth_sig = np.convolve(sig_extend,np.hamming(order),'same')
    smooth_sig = smooth_sig[int(order/2):(sig.size+int(order/2))]
    smooth_sig = np.amax(sig)/np.amax(smooth_sig)*smooth_sig

    # Plot signal for debug
    if(plot_show == 1):
        fig, ax = plt.subplots(ncols=2)
        ax[0].plot(sig)
        ax[0].plot(smooth_sig,'-r')
        ax[0].plot(med_sig,'black')
        ax[1].loglog(rfft(sig))
        ax[1].loglog(rfft(smooth_sig),'-r')
        ax[1].loglog(rfft(med_sig),'black')
        plt.show()
        
    return smooth_sig
 

def resampling(x,y,scale):
	l = x.shape
	
	u = np.arange(0, l[0])
	unew = np.arange(0, l[0]-1+scale, scale)
	
	tckx = interpolate.splrep(u, x, s=0)
	tcky = interpolate.splrep(u, y, s=0)
	
	xnew = interpolate.splev(unew, tckx, der=0)
	xder = interpolate.splev(unew, tckx, der=1)
	
	ynew = interpolate.splev(unew, tcky, der=0)
	yder = interpolate.splev(unew, tcky, der=1)
	
	return [xnew, ynew, xder, yder]

def richar_diff(data, mode='wrap'):
	length = data.shape
	length = length[0]
	A = np.concatenate((data[2:length], data[0:2]))
	B = np.concatenate((data[1:length], [data[0]]))
	C = np.concatenate((data[(length-2):length], data[0:length-2]))
	D = np.concatenate(([data[(length-1)]], data[0:length-1]))
	der = ((C - A) + 8 * (B - D))/12
	if(mode == 'clip'):
		der[length-1] = (-data[length-1]+4*data[length-2]-3*data[length-3])/2
		der[length-2] = (-data[length-2]+4*data[length-3]-3*data[length-4])/2
		der[0] = -(-data[0]+4*data[1]-3*data[2])/2
		der[1] = -(-data[1]+4*data[2]-3*data[3])/2
	return der

def angle_norm(o_data, plot_show = 0):
	data = o_data.copy()
	length = data.shape
	length = length[0]
	det_thre = 0.3
	index_decrease = []
	index_increase = []
	
	for i in range(0,length-1):
        
		if(data[i]-data[i+1]>np.pi/2):
            		index_increase = np.append(index_increase,i+1)
        	if(data[i]-data[i+1]<-np.pi/2):
            		index_decrease = np.append(index_decrease,i+1)
	
	for i in index_decrease:
		data[i:length] = data[i:length]-np.pi
	for i in index_increase:
		data[i:length] = data[i:length]+np.pi
	
	data = data/(max(data)-min(data))*2*np.pi
	data = data-min(data)-np.pi
	if(data[length-1] < data[0]):
		data = -data
        
	if(plot_show == 1):
        	fig, ax = plt.subplots(ncols = 1)
        	ax.plot(o_data)
        	ax.plot(data,'-r')
        	plt.show()
    
	return data

def clustering(seg,der,x,y):
	MAX_NUMBER_THRE = 200
	MAX_DISTANCE = 2
	i = 0
	j = 0

	while(i<len(seg)):
		j = i+1
		while(j<len(seg)):
			if (seg[i].size < MAX_NUMBER_THRE or arc_distance(seg[i],seg[j],der[i],der[j],x,y) < MAX_DISTANCE):
				seg[i] = np.vstack((seg[i],seg[j]))
				der[i] = np.vstack((der[i],der[j]))
				del seg[j]
				del der[j]
			else:
				j = j + 1
		i = i+1
	return [seg,der]			

def arc_distance(seg1,seg2,der1,der2,x,y, plot_show = 0):
	seg = np.vstack((seg1,seg2))
	der = np.vstack((der1,der2))
    
	[x_c, y_c, r, distance] = parameter_estimation(seg, der, mode = 0)
        
	if (plot_show == 1):
		print ['center', x_c, y_c, 'radius:',r,'distance: ',distance]
		fig, ax = plt.subplots(ncols=1)
		ax.plot(x,y,linewidth=2, color='red')
		ax.plot(seg[:,0],seg[:,1],color='black',linewidth=4)
		ax.plot(x_c,y_c,'x',linewidth=2, color='red')
		ax.add_artist(plt.Circle((x_c,y_c),r))
		ax.axis('equal')
		ax.invert_yaxis()
		plt.show()
 
	return distance

def find_circles(s, d, plot_show = 0):
	circles = np.array([])
	DIS_THRE = 2 
	MIN_R = 5
	MAX_R = 60
    	PER_THRE = 0.1
	
	l = len(s)
	
	for i in range(0,l):
		seg = s[i]
		der = d[i]
        
		[x_c, y_c, r, distance] = parameter_estimation(seg, der, mode = 0)	
		percentage = seg.size/2/np.pi/r/20
		
		if(distance<DIS_THRE and r <= MAX_R and r >= MIN_R and percentage >= PER_THRE):
			if(len(circles)>0):
				circles=np.vstack((circles,np.array([x_c,y_c,r])))
			else:
				circles=np.array([x_c,y_c,r])
       
		if(plot_show == 1):
        		fig, ax = plt.subplots(ncols=1)
			ax.plot(seg[:,0],seg[:,1],color='black')
			ax.plot(x_c,y_c,'x',linewidth=2, color='red')
            		ax.add_artist(plt.Circle((x_c,y_c),r))
			ax.axis('equal')
			ax.invert_yaxis()
			plt.show()
		
           
	return circles
	
def combine_circle(perimeters, thre):
    del_index = []
    for i in range(0,len(perimeters)):
        for j in range(i+1,len(perimeters)):
            [xi,yi,ri] = perimeters[i]
            [xj,yj,rj] = perimeters[j]
            dis = sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))
            if (dis<=thre):
                if (ri >= rj):
                    del_index.append(j)
                else:
                    del_index.append(i)
                    
    return np.delete(perimeters,del_index,0)
    
def perimeter_exaction(image, original_image, plot_show = 0):
	MIN_PER_LENGTH = 35
	MAX_PER_LENGTH = 1E4

	contours = measure.find_contours(image, 0)
	perimeter = np.array([])

	thre_local_min = -0.06
	thre_local_max = 0.4
	freq_thre = 0.002 # hamming window width
    # mutiple curve with same circle situation
	for n, contour in enumerate(contours):
            	[c_length,c_width] = contour.shape
		seg = []
		seg_der = []
		if(c_length >= MIN_PER_LENGTH and c_length <= MAX_PER_LENGTH):
		
			scale = 0.1
			[x, y, xder, yder] = resampling(contour[:,1], contour[:,0], scale)
			
			x=iff_filter(x, freq_thre)
			y=iff_filter(y, freq_thre)	
			data = np.concatenate((x,y),0)
			length = data.shape
			data = data.reshape(2,length[0]/2)
			data = data.T
			
			der = richar_diff(data,'wrap');
				
			fai = np.arctan(der[:,1]/(der[:,0]+np.spacing(1)))
			fai = angle_norm(fai)

			dfai = richar_diff(fai.T,'clip')
			
			dfs = np.true_divide(dfai,np.sqrt(der[:,1]*der[:,1]+der[:,0]*der[:,0])+np.spacing(1))

			p = np.arange(0, c_length-1+scale, scale)
			
		#	index_local_max = signal.argrelmax(dfs)
			index_local_min = signal.argrelmin(dfs)
		#	index_local_max = index_local_max[0]
			index_local_min = index_local_min[0]
			
			min_index = index_local_min[dfs[index_local_min] < thre_local_min]			
		#	max_index = index_local_max[dfs[index_local_max] > thre_local_max]
			
			c_index = min_index	
		#	c_index = np.append(min_index, max_index)
				
			if(c_index.size > 1):				
				tmpx = x[np.append(np.arange(c_index[c_index.size-1],x.size), np.arange(0,c_index[0]))]	
				tmpy = y[np.append(np.arange(c_index[c_index.size-1],x.size), np.arange(0,c_index[0]))]
				tmp_der = der[np.append(np.arange(c_index[c_index.size-1],x.size), np.arange(0,c_index[0])),:]
				tmp = np.array([tmpx,tmpy])
				seg.append(tmp.T)
				seg_der.append(tmp_der)
				for i in range(0,c_index.size-1):
					tmpx = x[np.arange(c_index[i],c_index[i+1])]	
					tmpy = y[np.arange(c_index[i],c_index[i+1])]
					tmp_der = der[np.arange(c_index[i],c_index[i+1]),:]
					tmp = np.array([tmpx,tmpy])
					seg.append(tmp.T)
					seg_der.append(tmp_der)
			else:
				tmp = np.array([x,y])
				seg.append(tmp.T)
				seg_der.append(der)
			
          		[seg, seg_der] = clustering(seg,seg_der,x,y)
           		circle_perimeter = find_circles(seg, seg_der)
            
			if(len(perimeter) == 0):
				perimeter = circle_perimeter
			elif (len(circle_perimeter)>0):
				perimeter = np.vstack((perimeter,circle_perimeter))
            
                	if (plot_show == 1):
                		x_max = max(contour[:,1])
				x_min = min(contour[:,1])
                		y_max = max(contour[:,0])
                		y_min = min(contour[:,0])
                		[y_limit, x_limit] = original_image.shape
                		if(x_min-10 >= 0):
                   			x_min = x_min-10
                		if(x_max+10 <= x_limit):
                   			x_max = x_max+10
                		if(y_min-10 >= 0):
                   			y_min = y_min-10
                		if(y_max+10 <= y_limit):
                   			y_max = y_max+10

                		image_part = original_image[y_min:y_max, x_min:x_max] 
			
	               		fig, [[ax, bx], [cx, dx]] = plt.subplots(2,2)
                		ax.imshow(image_part, cmap=plt.cm.gray)
                		ax.plot(contour[:, 1]-x_min+1, contour[:, 0]-y_min+1, linewidth=2,color='red')
                		ax.set_title('Original Image Edge')
                		bx.plot(x[0],y[0],'x',x[5],y[5],'o',x,y, linewidth=2, color='black')
                		bx.plot(x[c_index],y[c_index],'*', linewidth=3,color='red')
                		bx.axis('equal')
                		bx.set_title('Low Frequency Pass FIlter')
                		bx.invert_yaxis()
                		cx.plot(fai,linewidth=2, color='blue')
                		cx.plot(c_index,fai[c_index],'*')
                		cx.set_title('Angle Between Tangent and Axis')
                		dx.plot(dfs)
                		dx.plot(c_index,dfs[c_index],'*')
                		dx.set_title('Derivative of Curvature')
                		plt.show()

	return perimeter
	
def segmentation(image, method='otsu'):
	if (method == 'region'):
		sobel_image = filters.sobel(image)
        
		makers = sobel_image < sobel_image.max()*0.1
		makers = ndi.label(makers)[0]
	
		labels = watershed(sobel_image,makers) 
	
	elif (method == 'edge'):
		edges = canny(image,3)
		fill = ndi.binary_fill_holes(edges)
		labels = remove_small_objects(fill,10)
	
	elif (method == 'self_design'):
		width = 100;
		scale = 0.72;		
		[m, n] = image.shape
		thre = np.zeros((m,n))
		
		for i in range(0,n):
			ind_s = max(0,int(np.ceil(i-width/2)));
			ind_e = min(n-1,ind_s+width);
			current_image  = image[0:m-1, ind_s:ind_e]
			thre[0:m-1, i] = filters.threshold_otsu(current_image)*0.8


		labels = (image - thre) >=0	
	
	elif (method == 'thre_cons'):
		global_thre = image.max() * 0.3
		labels = image > global_thre

	elif (method == 'global_otsu'):
		global_thre = filters.threshold_otsu(image)
		labels = image > global_thre
	
	elif (method == 'local_otsu'):
		selem=disk(80)
		local_otsu = filters.rank.otsu(image, selem)
		labels = image > (np.true_divide(local_otsu,255))
	
	elif (method == 'yen'):
		global_thre = filters.threshold_yen(image)
                labels = image > (global_thre*2.5)
	
	elif (method == 'li'):
		global_thre = filters.threshold_li(image)
                labels = image > global_thre
	
	elif (method == 'isodata'):	 
		global_thre = filters.threshold_isodata(image)
                labels = image > global_thre
	
	elif (method == 'adaptive'):
		block_size = 100
		image = np.true_divide(image,image.max()+np.spacing(1)) * 255
		labels = filters.threshold_adaptive(image, block_size, offset=10)

	elif (method == 'R_Walker'):
		data = image + 0.35 * np.random.randn(*image.shape)
		markers = np.zeros(data.shape, dtype=np.uint)
		markers[data < -0.3] = 1
		markers[data > 1.3] = 2
		
		labels = random_walker(data, markers, beta=10, mode='cg_mg')
	
	return labels

def count_bubble(image, ref):
	# Constants
	Radius_Template        = 25
	Sigma_Gaussian_Filter  = 2
	Sigma_Canny_Filter     = 5
	Window_Size            = 5
        	
	part_image, region, angle = pre.interest_region(image, plot_image = 0)
	ref_rotate = rotate(ref,angle)
	part_ref = ref_rotate[region[0]:region[1], region[2]:region[3]]
	
	pre_image = pre.noise_reduction(part_image, part_ref, Window_Size, mode = 0)

	seg_image = segmentation(pre_image,'self_design')	
	
	perimeters = perimeter_exaction(seg_image,part_image)
	
	return len(perimeters)

#def count_bubble(image):
#    part_image, region, angle = pre.interest_region(image, plot_image = 0)
#    edge =  canny(part_image, sigma = 3)
#    perimeters = perimeter_exaction(edge,part_image)
#    return len(perimeters)

def main():
    try:
	# Default Parameters
	if len(sys.argv) <= 1:
		print 'Error: Filename Required'  
	if len(sys.argv) == 2:
		print 'Error: Background Filename Required'
	if len(sys.argv) >= 3:

	    # Constants
	    Radius_Template        = 25
	    Sigma_Gaussian_Filter  = 2
	    Sigma_Canny_Filter     = 5
	    Window_Size            = 5
	    Combine_Thre           = 15
	    
	    # Load Image
	    start_time = time.time()
	    print 'Reading the image: %s' % sys.argv[1]
	    
	    image_name = sys.argv[1]
	    ref_name = sys.argv[2]	
	
            image = rgb2gray(io.imread(sys.argv[1]))
	    ref = rgb2gray(io.imread(sys.argv[2]))
        
	    elapse_time = (time.time()-start_time)
            print 'Pre Processing .. %s' % elapse_time
	
	
	    part_image, region, angle = pre.interest_region(image, plot_image = 0)
	    ref_rotate = rotate(ref,angle)
	    part_ref = ref_rotate[region[0]:region[1], region[2]:region[3]]
	
	    pre_image = pre.noise_reduction(part_image, part_ref, Window_Size, mode = 0)

	    seg_image=segmentation(part_image,'self_design')	
	
	    edge =  canny(part_image, sigma = 3)
        
	    contours = measure.find_contours(seg_image, 0)
	    perimeters = perimeter_exaction(seg_image,part_image, plot_show = 0)
	    perimeters = combine_circle(perimeters,Combine_Thre)
        
	    fig, ax = plt.subplots(2,3)
	    ax[0,0].imshow(image,cmap=plt.cm.gray)
	    ax[0,0].set_title('Original Image')
	    ax[0,1].imshow(part_image,cmap=plt.cm.gray)
	    ax[0,1].set_title('Object Extaction')
	    ax[0,2].imshow(pre_image,cmap=plt.cm.gray)
	    ax[0,2].set_title('Median Filter and Denosing')
	    ax[1,0].imshow(seg_image,cmap=plt.cm.gray)
	    ax[1,0].set_title('Segmentation')
	    for n, contour in enumerate(contours):
		ax[1,1].plot(contour[:,1],contour[:,0],color='black')
	    ax[1,1].axis('equal')
	    ax[1,1].invert_yaxis()
	    ax[1,1].set_title('Edges')
	    ax[1,2].imshow(part_image,cmap=plt.cm.gray)
	    ax[1,2].set_title('Circle Recognition')
	    for i in range(0,len(perimeters)):
		[x, y, r] = perimeters[i]
		c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False) 
		ax[1,2].add_patch(c)
	    plt.show()	
	    print 'Succesfully Analized Image File: %s' % sys.argv[1]

    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()
