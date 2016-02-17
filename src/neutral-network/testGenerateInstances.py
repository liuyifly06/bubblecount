import sys, traceback, time
import numpy as np

from matplotlib import pyplot as plt

from skimage import io

def generateInstances(img, instanceSize, step):
	print 'Generating training instances ... instanceSize: ' \
		+ str(instanceSize) + ' step: ' + str(step)

	[m,n,c] = img.shape

	Y = np.arange(np.floor(instanceSize/2), (m - np.ceil(instanceSize/2)), step)
	X = np.arange(np.floor(instanceSize/2), (n - np.ceil(instanceSize/2)), step)
	
	totalInstancesY = len(Y)
	totalInstancesX = len(X)
	
	progressRatePrevious = 0

	Instances = np.zeros( (instanceSize ** 2 * c, totalInstancesX * totalInstancesY ))
	Labels = np.zeros( totalInstancesX * totalInstancesY )

	ind = -1
	
	start_time = time.time()

	for i in Y:   
    		for j in X:
        		currentY = i
        		currentX = j
        
        		boundaryT = currentY - np.floor(instanceSize/2)
        		boundaryD = currentY + np.floor(instanceSize/2)
        		boundaryL = currentX - np.floor(instanceSize/2)
        		boundaryR = currentX + np.floor(instanceSize/2)
        
        		ind = ind + 1
        
        		temp = img[boundaryT : boundaryD, boundaryL : boundaryR, :]

        		Instances[:, ind] = np.reshape(temp,(instanceSize ** 2 * c, 1)).T
         
    		elapse_time = time.time() - start_time
   		progressRate = np.true_divide(ind, totalInstancesX*totalInstancesY) * 100
    		remain_time = np.true_divide(elapse_time, progressRate) * 100 - elapse_time
    		progressRate = np.floor(np.true_divide(progressRate, 10))
    		if (progressRate > progressRatePrevious):
        		print str(progressRate*10) + \
            		'% test instances created, remaining time: ' +\
            		time.strftime(" %H:%M:%S", time.gmtime(remain_time))
    		progressRatePrevious = progressRate

	return [Instances, totalInstancesY, totalInstancesX]


def main():
    try:
	test_file = 'detector_7_no_5_angle_2.jpg'
	test = io.imread(test_file)
	instanceSize = 10;
	step = 2;
    	generateInstances(test, instanceSize, step)

    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()
