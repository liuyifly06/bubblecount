import numpy as np
import bubblecount.globalvar as gv
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
from bubblecount.progressbar import progress

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

def interest_region(image, plot_image = 0, lid_thickness = 680, \
                      bot_thickness = 230, side_thickness = 80, \
                    th_x = 0.5, th_y = 0.5):

    rotation_limit = 2.5 
    angle_u_lim = (90.0 + rotation_limit) / 180.0 * np.pi    
    angle_d_lim = (90.0 - rotation_limit) / 180.0 * np.pi
    
    #Grayscale Conversion and Edge Detection
    if(len(image.shape)>2):    
        img_edge = canny(rgb2gray(image), sigma = 3)
    else:
        img_edge = canny(image, sigma = 3) 
    
    #Image Rotation
    h, theta, d = hough_line(img_edge)
    h_peak, theta_peak, d_peak = hough_line_peaks(h, theta, d)    
    
    
    theta_rotation = theta_peak[np.where(np.absolute(theta_peak) \
                                         <= angle_u_lim)]
    theta_rotation = theta_rotation[np.where(np.absolute(theta_rotation) \
                                         >= angle_d_lim)]
    theta_rotation[np.where(theta_rotation > 0)] = \
             theta_rotation[np.where(theta_rotation > 0)] - np.pi 

    #print theta_rotation
        
    rotate_angle = 0
    if(theta_rotation.size):
        rotate_angle = np.pi/2 + np.mean(theta_rotation)
    rotate_angle = rotate_angle / 2.0
    img_rotate = rotate(img_edge, rotate_angle*180)

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
    
    if(len(temp) > 1000):        
        left = np.amin(temp)+lid_thickness
        right = np.amax(temp)-bot_thickness
        if (right <= left):
            left = np.amin(temp)+int(lid_thickness/2)
            right = np.amax(temp)+int(lid_thickness/2)
    else:
        left = 1000 + lid_thickness
        right = 3600 - bot_thickness

        bottom = bottom + side_thickness
        top = top - side_thickness
        
    image_rotation = rotate(image, rotate_angle*180)        
    interest_image = image_rotation[int(bottom):int(top), int(left):int(right)]
    
    if(plot_image == 1):
        fig, ax = plt.subplots(2,3)
        ax[0,0].imshow(image,cmap=plt.cm.gray)
        ax[0,0].set_title('Original Image')
        ax[0,1].imshow(img_edge,cmap=plt.cm.gray)
        ax[0,1].set_title('Canny Endge Detection')
        ax[0,2].imshow(image, cmap=plt.cm.gray)
        rows = image.shape[0]
        cols = image.shape[1]

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
        plt.setp(axHistx.get_xticklabels() + \
                 axHisty.get_yticklabels(), visible=False)
        
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
        ax[1,2].imshow(interest_image, cmap=plt.cm.gray)
        ax[1,2].set_title('Edge Detection')    
        plt.show()
        #"""

    return [interest_image, [bottom,top,left,right], rotate_angle*180]

def test(bubble_filename, background_filename, Window_Size = 5, plot_image = 1,
         pars = [1, 600, 50, 0, 0.5, 0.5]):
    # Load Image
    print 'Start reading images: \n    ' + bubble_filename
    image = io.imread(gv.__DIR__ + gv.pp__image_dir + bubble_filename)
    print '    ' +  background_filename
    background = io.imread(gv.__DIR__ + gv.pp__image_dir + background_filename)
    print 'Pre processing .. '  
    pre_image = noise_reduction(image, background, Window_Size, 0)
    interest_image = interest_region(image, plot_image = pars[0], 
        lid_thickness = pars[1], bot_thickness = pars[2],
        side_thickness = pars[3], th_x = pars[4], th_y = pars[5])

    if(plot_image == 1):
        ## plot image
        fig, ax = plt.subplots(ncols = 3)
        ax[0].imshow(image)
        ax[0].set_title('Original Image')
        ax[1].imshow(pre_image)
        ax[1].set_title('Noise Reduction')
        ax[2].imshow(interest_image[0])
        ax[2].set_title('Object Extaction')
        plt.show()

def batch_process(pars = [0, 630, 50, 40, 0.5, 0.5], label = False):
    f_read = open(gv.__DIR__ + gv.pp__image_dir + 'positiveInstances.dat', 'r')
    f_write = open(gv.__DIR__+ gv.pp__result_dir+ 'positiveInstances.dat', 'w')
    num_lines = sum(1 for line in f_read)
    f_read.seek(0)
    probar = progress.progress(0, num_lines)
    for i, line in enumerate(f_read):
        line_list = line.split()
        image_file = line_list[0]
        ref_file   = image_file[0:15] + '_background.jpg'
        ref_save_name = image_file[0:23] + '_background.jpg'
        probar.setCurrentIteration(i+1)
        probar.setInfo(prefix_info = 'Preprocessing ...',
                       suffix_info = image_file)
        probar.printProgress()

        image = io.imread(gv.__DIR__ + gv.pp__image_dir + image_file)
        ref   = io.imread(gv.__DIR__ + gv.pp__image_dir + ref_file)

        [interest_image, margin, angle] = interest_region(image,
                 plot_image = pars[0], lid_thickness = pars[1],
                 bot_thickness = pars[2], side_thickness = pars[3],
                 th_x = pars[4], th_y = pars[5])
        # generate background
        ref = rotate(ref, angle)
        ref = ref[int(margin[0]):int(margin[1]), int(margin[2]):int(margin[3])]

        # rotation of labeling box
        origin_y = (image.shape[0]-1)/2
        origin_x = (image.shape[1]-1)/2
        
        angle = -angle/180*3

        rotation_matrix = [[np.cos(angle), -np.sin(angle)],\
                           [np.sin(angle),  np.cos(angle)]]
        #print 'rotation: ' + str(rotation_matrix) 
        num= 0
        regions = []
        #print margin
        #print interest_image.shape
        for i in range(int(line_list[1])):
           x1 = int(line_list[4*i+2])
           y1 = int(line_list[4*i+3])
           x2 = x1 + int(line_list[4*i+4])
           y2 = y1 + int(line_list[4*i+5])
           
           rotated_box = np.dot(rotation_matrix, \
                                np.array([[x1-origin_x, x2-origin_x], \
                                          [y1-origin_y, y2-origin_y]])) 
           [[x1, x2], [y1, y2]] = np.add(rotated_box.astype(int),
              np.array([[origin_x, origin_x],[origin_y, origin_y]]))
           
           if (x1 >= margin[2] and x1 <= margin[3] and\
               y1 >= margin[0] and y1 <= margin[1]) or\
              (x2 >= margin[2] and x2 <= margin[3] and\
               y2 >= margin[0] and y2 <= margin[1]) :

               new_x1 = int(min(max(x1 - margin[2], 0), margin[3]-margin[2]))
               new_y1 = int(min(max(y1 - margin[0], 0), margin[1]-margin[0]))
               new_x2 = int(min(max(x2 - margin[2], 0), margin[3]-margin[2]))
               new_y2 = int(min(max(y2 - margin[0], 0), margin[1]-margin[0]))
               
               minmium_size = 5
               if(new_x2 - new_x1 >= minmium_size and \
                  new_y2 - new_y1 >= minmium_size):
                   regions.append(new_x1)
                   regions.append(new_y1)
                   regions.append(new_x2 - new_x1)
                   regions.append(new_y2 - new_y1)
                   num = num + 1
                   if(label):
                       interest_image[new_y1:new_y2, new_x1:new_x2, 0] = 1

        io.imsave(gv.__DIR__+ gv.pp__result_dir + ref_save_name, ref) 
        io.imsave(gv.__DIR__+ gv.pp__result_dir + image_file, interest_image) 
        f_write.write(image_file)
        f_write.write(' ' + str(num))
        
        for data in regions:
            f_write.write(' ' + str(data))
        
        f_write.write('\n')

    f_read.close()
    f_write.close()
