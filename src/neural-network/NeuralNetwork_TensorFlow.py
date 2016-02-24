# NeuralNetwork with google machine learning tool TensorFlow
import sys, traceback, time
import numpy as np
import tensorflow as tf
import trainGenerateInstances as trgi
import testGenerateInstances as tegi
import trainDataSet as trainds
import testDataSet as testds
from matplotlib import pyplot as plt
from skimage import io
from skimage import img_as_float
from skimage.filters import gaussian_filter

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
	start = 1
	
	batch_num = 2000
	batch_single = 1000
	sess = tf.Session()
	#training data
	trainDS = trainds.read_data_sets(ins_size, step);

	#testing  data
	testfile = '../../images/detector_2_no_5_angle_2.jpg'
	img = io.imread(testfile)
	testDS = testds.read_data_sets(img, ins_size, step)

	# Create the model
	x = tf.placeholder("float32", shape=[None, ins_size**2*3], name = "x-input")
	W = tf.Variable(tf.zeros([ins_size**2*3,label_option]), name = "weights")
	b = tf.Variable(tf.zeros([label_option]), name = "bias")
	p = tf.Variable(tf.zeros([batch_num, batch_single], name = "probability"))

	with tf.name_scope("Wx_b") as scope:
		y = tf.nn.softmax(tf.matmul(x, W) + b)	

	# Add summary ops to collect data
	tf.histogram_summary("weights", W)
	tf.histogram_summary("biases", b)
	tf.histogram_summary("y", y)

	# Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, label_option], name="y-input")	
		
	with tf.name_scope("xent") as scope:
  		cross_entropy = -tf.reduce_sum(y_*tf.log(y))
  		tf.scalar_summary("cross entropy", cross_entropy)
	with tf.name_scope("train") as scope:
  		train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
	with tf.name_scope("test") as scope:
  		probability = tf.to_float(tf.argmax(y,1))

		#test_image = tf.reshape(tf.convert_to_tensor(img, dtype=tf.float32), [1, img.shape[0], img.shape[1], 3])
		#test_result = tf.reshape(p, [1, testDS.test.ylength, testDS.test.xlength, 1])	
		#original_image = tf.Variable(tf.truncated_normal([1, img.shape[0], image.shape[1], 3]))
		#tf.image_summary("Original Test Image", test_image, max_images=1)
		#tf.image_summary("Test Result", test_result, max_images=1)
	
	
	merged = tf.merge_all_summaries()
	writer = tf.train.SummaryWriter("/tmp/neuralnetwork_logs", sess.graph_def)
	init = tf.initialize_all_variables()	
	sess.run(init)
	for i in range(batch_num):
    	    batch_xs, batch_ys = trainDS.train.next_batch(batch_single)
    	    feed = {x: batch_xs, y_: batch_ys}
    	    sess.run(train_step, feed_dict=feed)
	    p[i,:] = probability;
	    img_res = tf.reshape(p,[batch_num*batch_single/testDS.test.xlength,testDS.test.xlength])  
	    feed = {x: testDS.test.next_batch(batch_single), y_: np.zeros((batch_single,label_option))}
    	    result = sess.run([merged, img_res], feed_dict=feed)    
            summary_str = result[0]
    	    writer.add_summary(summary_str, i)
	
	#fig, ax = plt.subplots(ncols = 1)
	#ax.imshow(np.asarray(sess.run(img_res.eval()),feed_dict=feed))) #how to plot a tensor ?#
	#plt.show()

	sess.close()
	
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()
