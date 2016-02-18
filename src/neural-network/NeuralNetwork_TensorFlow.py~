# NeuralNetwork with google machine learning tool TensorFlow
import sys, traceback, time
import numpy as np
import tensorflow as tf
import trainGenerateInstances as trgi
import testGenerateInstances as tegi

from matplotlib import pyplot as plt
from skimage import io
from skimage.filters import gaussian_filter

from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData

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

def displayTime(str_process ,elap_time, rema_time):
	if(rema_time < 0):
		print	'Current process : ' + str_process + \
             		'|| Time Elapsed : ' + \
			time.strftime(" %H:%M:%S", time.gmtime(elap_time))
	else:
		print	'Current process : ' + str_process + \
             		'|| Time Elapsed : ' + \
			time.strftime(" %H:%M:%S", time.gmtime(elap_time)) + \
              		'|| Time Remaining : ' + \
			time.strftime(" %H:%M:%S", time.gmtime(rema_time))

def main():
    try:
        init_time = time.time()
	ins_size = 20
	step = 10
	label_option = 100
	
	#training data
	displayTime('Generating Data', time.time()-init_time, -1)	
	[x_data, y_data] = trgi.generateInstancesNN(ins_size, step, plot_show = 0)	
	
	#testing data
	testfile = 'detector_2_no_5_angle_2.jpg'
	img = io.imread(testfile)
	[x_test, len_y, len_x] = tegi.generateInstances(img, ins_size, step)
	
	x = tf.placeholder("float32", shape=[None, ins_size**2*3])
	y_ = tf.placeholder(tf.float32, [None, label_option])	

	W = tf.Variable(tf.zeros([ins_size**2*3,label_option]))
	b = tf.Variable(tf.zeros([label_option]))
	y = tf.nn.softmax(tf.matmul(x, W) + b)	
		
			
	
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
	
	init = tf.initialize_all_variables()	
	sess = tf.Session()
	sess.run(init)

	# training
		
#	for i in range(1000):
#  		batch_xs, batch_ys = mnist.train.next_batch(100)
#  		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	sess.run(train_step, feed_dict={x: x_data.T, y_: y_data.T})
	res = tf.argmax(y,1)
	print sess.run(res)
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#	print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

	sess.close()
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()
