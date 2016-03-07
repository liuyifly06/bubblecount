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
		
	ins_size = 20
	step = 10
	label_option = 100
	
	sess = tf.Session()
	
	#training data
	trainDS = trainds.read_data_sets(ins_size, step);

	#testing  data
	testfile = '../../images/detector_2_no_5_angle_2.jpg'
	img = io.imread(testfile)
	testDS = testds.read_data_sets(img, ins_size, step)
	
	batch_num = testDS.test.ylength
	batch_size = testDS.test.xlength

	# Create the model
	x = tf.placeholder(tf.float32, shape=[None, ins_size**2*3], name = "x_input")
	y_ = tf.placeholder(tf.float32, [None, label_option], name="y_input") 
	batch_index = tf.placeholder(tf.int32, shape = [], name = "batch_index")
	W = tf.Variable(tf.zeros([ins_size**2*3,label_option]), name = "weights")
	b = tf.Variable(tf.zeros([label_option]), name = "bias")
	p = tf.Variable(tf.ones([1,batch_size]), name = "p_batch")

	with tf.name_scope("Wx_b") as scope:
		y = tf.nn.softmax(tf.matmul(x, W) + b)
	with tf.name_scope("test") as scope:
  		p_batch = tf.reshape(tf.to_float(tf.argmax(y,1)),[1,batch_size])
		p = tf.cond(batch_index > 0, lambda: tf.concat(0, [p, p_batch]), lambda: p_batch)
		paddings = tf.constant([[0, batch_num-1-batch_index], [0, 0]])
		p_all = tf.pad(p,paddings)

	# Add summary ops to collect data
	#tf.histogram_summary("weights", W)
	#tf.histogram_summary("biases", b)
	#tf.histogram_summary("y", y)

	# Define loss and optimizer
	with tf.name_scope("xent") as scope:
  		cross_entropy = -tf.reduce_sum(y_*tf.log(y))
  		tf.scalar_summary("cross entropy", cross_entropy)
	with tf.name_scope("train") as scope:
  		train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

	#Image Summary:
	test_image = tf.reshape(tf.convert_to_tensor(img, dtype=tf.float32), [1, img.shape[0], img.shape[1], 3])
	test_result = tf.reshape(p_all, [1, batch_num, batch_size, 1])	
	original_image = tf.Variable(tf.truncated_normal([1, img.shape[0], img.shape[1], 3]))
	tf.image_summary("Original Test Image", test_image, max_images=1)
	tf.image_summary("Test Result", test_result, max_images=1)
	
	merged = tf.merge_all_summaries()
	writer = tf.train.SummaryWriter("/tmp/neuralnetwork_logs", sess.graph_def)
	init = tf.initialize_all_variables()	
	sess.run(init)

	for i in range(batch_num):
    	    batch_xs, batch_ys = trainDS.train.next_batch(batch_size)
    	    feed = {x: batch_xs, y_: batch_ys}
    	    sess.run(train_step, feed_dict=feed)
	
	for i in range(batch_num):
	    feed = {x: testDS.test.next_batch(batch_size), batch_index: i, y_: batch_ys}
	    result = sess.run([merged, test_result], feed_dict=feed)
            summary_str = result[0]
    	    writer.add_summary(summary_str, i)
	"""
	for i in range(batch_num):
	    feed = {x: testDS.test.next_batch(batch_size), batch_index: i}
    	    result = sess.run([merged, test_result], feed_dict=feed)
            summary_str = result[0]
    	    writer.add_summary(summary_str, i)
	"""
	
	sess.close()
	
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()
