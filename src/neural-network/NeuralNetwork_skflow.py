# NeuralNetwork with google machine learning tool TensorFlow
# Mutiple layer
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
import random
import skflow
from sklearn import datasets, cross_validation, metrics

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
      random.seed(42) 
      ins_size = 20
      step = 4
      label_option = 100
    
      #training data
      trainDS = trainds.read_data_sets(ins_size, step, label_option);

      classifier = skflow.TensorFlowDNNClassifier(
        hidden_units =[500,1000,500],
        n_classes = label_option,
        batch_size = 200000,
        steps = 1,
        learning_rate = 0.01)

      classifier.fit(trainDS.train.images, np.argmax(trainDS.train.labels, axis = 1))
   
      data_filename = 'skflow.txt'
      start_time = time.time()
      number = np.zeros((41,4))
      index = -1

      for det in range(1,8):
            if (det!=6):
                num_n = 6
            else:
                num_n = 5
            for n in range(0,num_n):
                index = index + 1
                temp = np.zeros((3,1))
                for angle in range(1,4):
                    filename = '../../images/detector_' + str(det) + '_no_' + str(n) \
                                    + '_angle_' + str(angle) + '.jpg'

                    elapse_time = (time.time()-start_time)
                    if(elapse_time >= 1):
                        remain_time = elapse_time/(index*3+angle-1)*41*3-elapse_time
                        print 'Processing .. ' + filename \
                                +  time.strftime(" %H:%M:%S", time.gmtime(elapse_time)) \
                                + ' has past. ' + 'Remaining time: ' \
                                +  time.strftime(" %H:%M:%S", time.gmtime(remain_time))
                    else:
                        print 'Processing .. ' + filename \
                                +  time.strftime(" %H:%M:%S", time.gmtime(elapse_time)) \
                                + ' has past'

                    img = io.imread(filename)
                    testDS = testds.read_data_sets(img, ins_size, step)
                    batch_num  = testDS.test.ylength
                    batch_size = testDS.test.xlength
		    y = classifier.predict(testDS.test.images)
                    temp[angle-1] = np.sum(y)
                number[index,1] = np.mean(y)
                number[index,2] = np.std(y)

      manual_count = np.array([1,27,40,79,122,160,1,18,28,42,121,223,0,11,24,46,\
                142,173,3,19,23,76,191,197,0,15,24,45,91,152,0,\
                16,27,34,88,0,9,12,69,104,123]) 
      number[:,0] = manual_count.T
      number.tofile(data_filename,sep=" ")

      sess = tf.Session();
      
      _y = tf.placeholder(tf.float32, [None], name="y")
      p = tf.reshape(tf.to_float(_y),[batch_num, batch_size])
      test_result = tf.reshape(p, [1, batch_num, batch_size, 1])
      _test_label  = tf.image_summary("Test Label", test_result, max_images=1)
      merged_image = tf.merge_summary([_test_label])
      writer = tf.train.SummaryWriter("/tmp/neuralnetwork_logs", \
                                      sess.graph.as_graph_def(add_shapes=True))
      init = tf.initialize_all_variables()
      sess.run(init)
      
      feed = {_y:y}
      result_image = sess.run([merged_image, test_result], feed_dict=feed) 
      writer.add_summary(result_image[0])
      sess.close()
    
    except KeyboardInterrupt:
      print "Shutdown requested... exiting"
    except Exception:
      traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()
