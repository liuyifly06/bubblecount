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
        print    'Current process : ' + str_process + \
                     '|| Time Elapsed : ' + \
            time.strftime(" %H:%M:%S", time.gmtime(elap_time))
    else:
        print    'Current process : ' + str_process + \
                     '|| Time Elapsed : ' + \
            time.strftime(" %H:%M:%S", time.gmtime(elap_time)) + \
                      '|| Time Remaining : ' + \
            time.strftime(" %H:%M:%S", time.gmtime(rema_time))

def main():
    try:    
      ins_size = 20
      step = 4
      label_option = 100

      sess = tf.Session()
    
      #training data
      trainDS = trainds.read_data_sets(ins_size, step);

      #testing  data
      testfile = '../../images/detector_2_no_5_angle_2.jpg'
      img = io.imread(testfile)
      testDS = testds.read_data_sets(img, ins_size, step)
    
      batch_num  = testDS.test.ylength
      batch_size = testDS.test.xlength

      # Create the model
      x = tf.placeholder(tf.float32, shape=[None, ins_size**2*3], name = "x_input")
      y_ = tf.placeholder(tf.float32, [None, label_option], name="y_input")
      batch_index = tf.placeholder(tf.int32, shape = [], name = "batch_index") 
      W = tf.Variable(tf.zeros([(ins_size**2*3),label_option]), name = "weights")
      b = tf.Variable(tf.zeros([label_option]), name = "bias")
      p_train = tf.Variable(tf.ones([batch_num,batch_size]), name = "probability")
      p_test  = tf.Variable(tf.ones([batch_num,batch_size]), name = "probability")

      with tf.name_scope("Wx_b") as scope:
        y = tf.nn.softmax(tf.matmul(x, W) + b)

      #Add summary ops to collect data
        w_hist = tf.histogram_summary("weights", W)
        b_hist = tf.histogram_summary("biases", b)
        y_hist = tf.histogram_summary("y", y)

      # Define loss and optimizer
      with tf.name_scope("xent") as scope:
        cross_entropy = -tf.reduce_sum(y_*tf.log(y))
        ce_summ = tf.scalar_summary("cross entropy", cross_entropy)
      with tf.name_scope("train") as scope:
        p_train_batch = tf.reshape(tf.to_float(tf.argmax(y_,1)),[1, batch_size])
        update_op_train = tf.scatter_update(p_train, 
                      tf.expand_dims(batch_index%batch_num, 0),p_train_batch)        
        train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)
      with tf.name_scope("test") as scope:
        p_test_batch = tf.reshape(tf.to_float(tf.argmax(y,1)),[1, batch_size])
        update_op_test = tf.scatter_update(p_test,
                      tf.expand_dims(batch_index, 0),p_test_batch)
                      #tf.expand_dims(p_test_batch, 0)) 
        #Image Summary:
        train_label = tf.reshape(p_train, [1, batch_num, batch_size, 1])
        test_image  = tf.reshape(tf.convert_to_tensor(img, dtype=tf.float32), \
            [1, img.shape[0], img.shape[1], 3])
        test_result = tf.reshape(p_test, [1, batch_num, batch_size, 1])
        _train_label = tf.image_summary("Train Label", train_label, max_images=1) 
        _test_image  = tf.image_summary("Test Image", test_image, max_images=1)
        _test_label  = tf.image_summary("Test Label", test_result, max_images=1)
    
      merged_train = tf.merge_summary([ \
                     w_hist, b_hist, y_hist, ce_summ])
      merged_image = tf.merge_summary([ \
                     _train_label, _test_image, _test_label])

      writer = tf.train.SummaryWriter("/tmp/neuralnetwork_logs", \
                                      sess.graph.as_graph_def(add_shapes=True))
      init = tf.initialize_all_variables()
      sess.run(init)
      
      #train
      for i in range(batch_num*100):
        batch_xs, batch_ys = trainDS.train.next_batch(batch_size)
        feed = {x: batch_xs, y_: batch_ys, batch_index: i}
        sess.run([update_op_train, train_step], feed_dict=feed)
   
	if i % 100 == 0:  # Record summary data
          result = sess.run([merged_train, \
               update_op_train, train_step], feed_dict=feed)
          writer.add_summary(result[0], i)
        else:
          sess.run([update_op_train, train_step], feed_dict=feed)

      #sess.run(tf.Print(W,[W,b]))
      #test
      for i in range(batch_num):
        batch_xs = testDS.test.next_batch(batch_size)
        feed = {x: batch_xs, batch_index: i, y_: np.ones((batch_size, label_option))}
        result = sess.run(update_op_test, feed_dict=feed)
      
      result_image = sess.run(merged_image) 
      writer.add_summary(result_image, batch_num)
      sess.close()
    
    except KeyboardInterrupt:
      print "Shutdown requested... exiting"
    except Exception:
      traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()
