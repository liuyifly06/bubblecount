import sys, traceback
import numpy as np
import preprocessing as pre

from matplotlib import pyplot as plt

from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate


def main():
    try:
	if len(sys.argv) <= 1:
		print 'Error: Filename Required'  
	if len(sys.argv) == 2:
		print 'Error: Background Filename Required'
	if len(sys.argv) >= 3:

	    # Constants
	    Window_Size = 5
	    image_name = sys.argv[1]
	    ref_name = sys.argv[2]	
	
            image = rgb2gray(io.imread(sys.argv[1]))
	    ref = rgb2gray(io.imread(sys.argv[2]))

	    part_image, region, angle = pre.interest_region(image, plot_image = 0)
	    ref_rotate = rotate(ref,angle)
	    part_ref = ref_rotate[region[0]:region[1], region[2]:region[3]]
	
	    pre_image = pre.noise_reduction(part_image, part_ref, Window_Size, mode = 0)
	    io.imsave('pre_image.jpg',pre_image)

    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()
