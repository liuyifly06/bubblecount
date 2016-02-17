import sys, traceback, time
import numpy as np

from skimage.transform import rotate, hough_line, hough_line_peaks
from skimage import io, img_as_ubyte
from skimage.color import rgb2gray
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage.feature import canny
from skimage.filters import sobel
from skimage.filters.rank import median
from skimage.morphology import disk

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# analyzing RGB image through each channel 
@adapt_rgb(each_channel)
def median_each(image, selem):
    return median(image, selem)

# analyzing RGB image through value of HSV
@adapt_rgb(hsv_value)
def median_hsv(image, selem):
    return median(image, selem)

def noise_reduction(image, background, window_size = 5, mode = 0):
	#  GrayScale image use mode = 0
	#  mode = 0: dealing RGB image with each channel 
	#  mode!= 0: dealing RGB image with HSV value 
	if(mode == 0):
		med_image = median_each(image, disk(window_size))
		med_bckg  = median_each(background, disk(window_size))

	else :
		med_image = img_as_ubyte(median_hsv(image, disk(window_size)))
		med_bckg  = img_as_ubyte(median_hsv(background, disk(window_size)))

	norm_image = np.true_divide(med_image, (med_bckg/255*254+1))/255	
	return norm_image

def interest_region(image, plot_image = 0):
	# constant 
	th_x = 0.5
	th_y = 0.5
	lid_thickness = 680
	bot_thickness = 230
	side_thickness = 80

	rotation_limit = 2
	angle_u_lim = (90.0 + rotation_limit) / 180.0 * np.pi	
	angle_d_lim = (90.0 - rotation_limit) / 180.0 * np.pi
	
	#Grayscale Conversion and Edge Detection
	if(len(image.shape)>2):
		image = rgb2gray(image)
	img_edge =  canny(image, sigma = 3)
	
	#Image Rotation
	h, theta, d = hough_line(img_edge)
	h_peak, theta_peak, d_peak = hough_line_peaks(h, theta, d)	
	
	theta_rotation = theta[np.where(np.absolute(theta_peak)<=angle_u_lim)]
	theta_rotation = theta_rotation[np.where(np.absolute(theta_rotation)>=angle_d_lim)]

	if(theta_rotation.size):
		rotate_angle = np.pi/2-np.absolute(np.mean(theta_rotation))
		img_rotate = rotate(img_edge,rotate_angle*180)

	#rectangular region selection
	index = np.where(img_rotate>0)

	[hy,b] = np.histogram(index[0],100)
	by = (b[0:(len(b)-1)]+b[1:len(b)])/2
	[hx,b] = np.histogram(index[1],100)
	bx = (b[0:(len(b)-1)]+b[1:len(b)])/2
	
	temp = by[np.where(hy>=th_y*np.mean(hy))]
	if(len(temp)>0):
		bottom = np.amin(temp)
		top = np.amax(temp)	
	else:
		bottom = 450
		top = 1500

	temp = bx[np.where(hx>=th_x*np.mean(hx))]
    
	if(len(temp)>0):		
		left = np.amin(temp)+lid_thickness
		right = np.amax(temp)-bot_thickness
		if (right <= left):
			left = np.amin(temp)+int(lid_thickness/2)
			right = np.amax(temp)+int(lid_thickness/2)
	else:
		left = 1700
		right = 3600-bot_thickness

        bottom = bottom + side_thickness
        top = top - side_thickness
        
        #print [bottom,top,left,right];
	image_rotation = rotate(image,rotate_angle*180)	
	
	interest_image = image_rotation[bottom:top,left:right];
	
	if(plot_image == 1):
		fig, ax = plt.subplots(2,3)
		ax[0,0].imshow(image,cmap=plt.cm.gray)
		ax[0,0].set_title('Original Image')
		ax[0,1].imshow(img_edge,cmap=plt.cm.gray)
		ax[0,1].set_title('Canny Endge Detection')
		ax[0,2].imshow(image, cmap=plt.cm.gray)
		rows, cols = image.shape
		for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    			y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    			y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
    			ax[0,2].plot((0, cols), (y0, y1),'-r')
		ax[0,2].axis((0, cols, rows, 0))
		ax[0,2].set_title('Detected lines')
		ax[0,2].set_axis_off()
		ax[1,0].imshow(img_rotate,cmap=plt.cm.gray)
		ax[1,0].set_title('Image Edge after Rotation')
		ax[1,1].scatter(index[1],index[0])
		ax[1,1].set_aspect('equal')
		ax[1,1].set_title('Pixel Axises Histogram')
		divider = make_axes_locatable(ax[1,1])
		axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax[1,1])
		axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=ax[1,1])
		# make some labels invisible
		plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)
		# now determine nice limits by hand:
		axHistx.hist(index[1], bins=bx)
		axHistx.axhline(y=th_y*np.mean(hy),c="red",linewidth=2,zorder=0)
		axHisty.hist(index[0], bins=by, orientation='horizontal')
		axHisty.axvline(x=th_x*np.mean(hx),c="red",linewidth=2,zorder=0)
		#axHistx.axis["bottom"].major_ticklabels.set_visible(False)
		for tl in axHistx.get_xticklabels():
    			tl.set_visible(False)
		axHistx.set_yticks([0, 400, 800, 1200])
		#axHisty.axis["left"].major_ticklabels.set_visible(False)
		for tl in axHisty.get_yticklabels():
    			tl.set_visible(False)
		axHisty.set_xticks([0, 400, 800, 1200])
		ax[1,2].imshow(interest_image,cmap=plt.cm.gray)
		ax[1,2].set_title('Edge Detection')	
		plt.show()
		#"""
	return [interest_image,[bottom,top,left,right],rotate_angle*180]

def main():
    try:
	# Default Parameters
	if len(sys.argv) == 1:
		print 'Error: Filename Required'  
	if len(sys.argv) >= 2:
	
		# Constants
		Window_Size = 5
	
		# Load Image
		start_time = time.time()
		print 'Start reading the image: %s' % sys.argv[1]
		image_name = sys.argv[1]

       		image = io.imread(sys.argv[1])
		background = io.imread(sys.argv[2])
   
		elapse_time = (time.time()-start_time)
        	print 'Pre processing .. %s' % elapse_time

		pre_image = noise_reduction(image, background, Window_Size, 0)
		interest_image = interest_region(image)

		## plot image
		elapse_time = (time.time()-start_time)
        	print 'Start ploting image .. %s' % elapse_time

 		fig, ax = plt.subplots(ncols = 3)
		ax[0].imshow(image)
		ax[0].set_title('Original Image')
		ax[1].imshow(pre_image)
		ax[1].set_title('Noise Reduction')
		ax[2].imshow(interest_image)
		ax[2].set_title('Object Extaction')
		plt.show()
				
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()
