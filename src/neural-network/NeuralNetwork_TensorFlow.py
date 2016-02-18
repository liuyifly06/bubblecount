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
	ins_size = 10
	step = 2
	label_option = 100
	
	#training data
	displayTime('Generating Data', time.time()-init_time, -1)
	[x_data, y_data] = trgi.generateInstancesNN(ins_size, step, plot_show = 0)	

	#testing data
	testfile = '../../images/detector_2_no_5_angle_2.jpg'
	img = io.imread(testfile)
	[x_test, len_y, len_x] = tegi.generateInstances(img, ins_size, step)
	
	x = tf.placeholder("float32", shape=[None, ins_size**2*3])

	W = tf.Variable(tf.zeros([ins_size**2*3,label_option]))
	b = tf.Variable(tf.zeros([label_option]))
	y = tf.nn.softmax(tf.matmul(x, W) + b)	

	# placeholder for correct answer	
	y_ = tf.placeholder(tf.float32, [None, label_option])	
		
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
	
	init = tf.initialize_all_variables()	
	sess = tf.Session()
	sess.run(init)

	# training
	for i in range(1):
  		sess.run(train_step, feed_dict={x: x_data.T, y_: y_data.T})
	
	# testing
	probability = tf.argmax(y,1);
	img_res = tf.reshape(probability,[len_y,len_x])
	
	#show result	
	fig, ax = plt.subplots(ncols = 1)
	ax.imshow(np.sqrt(sess.run(img_res,feed_dict={x: x_test.T})))
	plt.show()

	sess.close()
	
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()
