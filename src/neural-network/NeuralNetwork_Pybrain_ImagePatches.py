import sys, traceback, time
import numpy as np
import preprocessing as pre
import trainGenerateInstances as trgi
import testGenerateInstances as tegi

from matplotlib import pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import gaussian_filter
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork, \
        LinearLayer, SigmoidLayer, FullConnection

from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData

def GenerateTrainDataSet(instanceSize, step):
	[ia, ta] = trgi.generateInstancesNN(instanceSize, step, plot_show = 0)
	ia = ia.T	
	ds = SupervisedDataSet(ia.shape[1], 1)
    	assert(ia.shape[0] == ta.shape[0])
  
	for i in range(0,ia.shape[0]):
		ds.addSample(ia[i,:], ta[i])
    	return ds

def GenerateTestDataSet(img, instanceSize, step):
	[ia, Y, X] = tegi.generateInstances(img, instanceSize, step)
	ia = ia.T
	ta = np.zeros(ia.shape[0])	
	ds = SupervisedDataSet(ia.shape[1], 1)
    	assert(ia.shape[0] == ta.shape[0])

	for i in range(0,ia.shape[0]):
		ds.addSample(ia[i,:], ta[i])
   	return [ds, Y, X]

def build_network(input_dim,layers,output_dim):
    
    n = FeedForwardNetwork()
    inLayer = LinearLayer(input_dim)
    hiddenLayer = SigmoidLayer(layers)
    outLayer = LinearLayer(output_dim)
    
    n.addInputModule(inLayer)
    n.addModule(hiddenLayer)
    n.addOutputModule(outLayer)
    
    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)
    
    n.addConnection(in_to_hidden)
    n.addConnection(hidden_to_out)
    n.sortModules()
    
    return n

def countBubbles(net, test_img, instanceSize, step, plot_show = 0):
	[TestDs, Y, X] = GenerateTestDataSet(test_img, instanceSize, step)
        
	res = np.zeros(X*Y) 
	index = -1
	for inp, targ in TestDs:
	    index = index + 1
            res[index] = net.activate(inp)
	
	img_res = np.reshape(res,(Y,X))
	img_filtered = gaussian_filter(img_res, sigma=0.4)
	
	if(plot_show == 1):
		fig, ax = plt.subplots(ncols = 3)
		ax[0].imshow(test_img)
		ax[0].set_title('Test Image')
		im1 = ax[1].imshow(img_res)
		ax[1].set_title('Labling Result')
		im2 = ax[2].imshow(img_filtered)
		ax[2].set_title('Labling Result After Gaussian Filter')
		fig.colorbar(im1, ax=ax[0])
		plt.show()
	return [np.sum(img_res), np.sum(img_filtered)]	

def main():
    try:
        start_time = time.time()
        elapse_time = time.time() - start_time
        print 'Start reading images to generate data ...' \
                +  time.strftime(" %H:%M:%S", time.gmtime(elapse_time)) \
                + ' has past. '

        #Read Data
	instanceSize = 10
	step = 2
	TrainDs = GenerateTrainDataSet(instanceSize, step)

        elapse_time = (time.time()-start_time)
        print 'Start training neutral neutwrok ...' \
                +  time.strftime(" %H:%M:%S", time.gmtime(elapse_time)) \
                + ' has past. '
        
        #Build Netwrok
        LAYERS_NUM = 3
        net = build_network(TrainDs['input'].shape[1],LAYERS_NUM,1)
        # net = buildNetwork(m*n, LAYERS_NUM, 1)
        
        #Training
        trainer = BackpropTrainer(net, dataset=TrainDs)
	t1 = (time.time()-start_time)
	print trainer.train()
	t2 = (time.time()-start_time)
	print 'Training Time for One Epoch %.1f s' % (t2-t1)
	"""
	error_past = 0
	error_now = 1
	error_thre = 1E-9
	num_epos = 10
	i = 0
	while (np.abs(error_now - error_past) > error_thre and i <= num_epos):
		i = i + 1;
		error_past = error_now
		error_now = trainer.train()
		print [error_now, error_past]
        trainer.trainUntilConvergence()#"""
        
	#Tesing
        elapse_time = (time.time()-start_time)
        print 'Start tesing instances ...' \
                +  time.strftime(" %H:%M:%S", time.gmtime(elapse_time)) \
                + ' has past. '
	
	begin_time = (time.time()-start_time)
	index = -1
	number = np.zeros((41,3))
	number_filtered = np.zeros((41,3))

	for det in range(1,8):
              	if (det!=6):
                    num_n = 6
                else:
                    num_n = 5
                for n in range(0,num_n):
                    index = index + 1
                    temp = np.zeros(3)
		    temp_filtered = np.zeros(3)
                    for angle in range(1,4):
                        filename = 'detector_' + str(det) + '_no_' + str(n) \
                                        + '_angle_' + str(angle) + '.jpg'

                        elapse_time = (time.time()-begin_time)
                        if((index*3+angle-1) >= 1):
                            remain_time = elapse_time/(index*3+angle-1)*41*3-elapse_time
                            print 'Tesing instances in ' + filename \
                                    +  time.strftime(" %H:%M:%S", \
					time.gmtime(time.time() - start_time)) \
                                    + ' has past. ' + 'Remaining time: ' \
                                    +  time.strftime(" %H:%M:%S", time.gmtime(remain_time))
                        else:
                            print 'Tesing instances in ' + filename \
                                    +  time.strftime(" %H:%M:%S", \
					time.gmtime(time.time() - start_time)) \
                                    + ' has past'
            
#			filename = 'detector_2_no_5_angle_1.jpg'
                        image = io.imread(filename)
            
                     	[temp[angle-1], temp_filtered[angle-1]] = \
				countBubbles(net, image, instanceSize, step, plot_show = 0)
			print [temp[angle-1], temp_filtered[angle-1]]
                    number[index,1] = np.mean(temp)
                    number[index,2] = np.std(temp)
                    number_filtered[index,1] = np.mean(temp_filtered)
                    number_filtered[index,2] = np.std(temp_filtered)
        
        manual_count = np.array([1,27,40,79,122,160,1,18,28,42,121,223,0,11,24,46,\
                    142,173,3,19,23,76,191,197,0,15,24,45,91,152,0,\
                    16,27,34,88,0,9,12,69,104,123])

        number[:,0] = manual_count.T
	number_filtered[:,0] = manual_count.T
	data_filename = 'neutralnetwork.txt'
	data_filename_filtered = 'neutralnetwork_filtered.txt'
        number.tofile(data_filename,sep=", ")
	number_filtered.tofile(data_filename_filtered,sep=", ")

    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()
