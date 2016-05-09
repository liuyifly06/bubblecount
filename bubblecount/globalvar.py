import os
#dirctory of python package
__DIR__ = os.path.dirname(os.path.realpath(__file__)) + '/..'

#Train Images
__TrainImage__ = [ \
'detector_1_no_5_angle_2.jpg', \
'detector_2_no_5_angle_3.jpg'
]
__TrainImageDir__ = '/images/AfterPreprocessing/'
#__TrainImageDir__ = '/images/original/'
#benchmark
ac__file_dir = '/bubblecount/benchmark/'

#preprocessing
pp__image_dir = '/images/original/'
pp__result_dir = __TrainImageDir__

#NeuralNetwork
tensorflow_log_dir = '/result/logs/'
#deeplearning
dp_test_save_data = True
dp_image_save = True
dp__image_dir = '/images/deepLearning/'
dp__result_filename = 'deepLearningResult.txt'
dp__accuracy_filename = 'deepLearningAccuracy.txt'
dp__tuningPar_dir = '/result/deepLearning/'
dp__tuningPar_filename = 'deepLearningParTune.dat'
#convolutional
cnn__image_dir = '/images/convolutionalNN/'
cnn__result_filename = 'convolutionalNNResult.txt'
cnn__accuracy_filename = 'convolutionalNNAccuracy.txt'

#curvature
cu__image_dir = '/images/curvature/'
cu__result_filename = 'curvatureResult.txt'

#show progress bar
show_progress = True
#training logs
log_write = True
#data set generation /neuralnetwork/dataset.py in generateInstancesNN()
ds_show_filename = True

#test
label_linearity_file_dir = '/result/benchmark/'
label_linearity_file = 'labellinearity.dat'

#Memory Limit
MEM_LIM = 1024*1024*1024
