import os

#dirctory of python package
__DIR__ = os.path.dirname(os.path.realpath(__file__)) + '/..'

#Train Images
__TrainImage__ = [ \
'detector_5_no_5_angle_2.jpg', \
'detector_7_no_5_angle_2.jpg'
]
__TrainImageDir__ = '/images/AfterPreprocessing/'

#preprocessing
pp__image_dir = '/images/original/'
pp__result_dir = __TrainImageDir__

#NeuralNetwork
tensorflow_log_dir = '/result/logs/'
#deeplearning 
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
