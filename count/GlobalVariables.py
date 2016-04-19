#dirctory of python package
__DIR__ = '/home/yi/Documents/bubble-counting'

#Train Images
__TrainImage__ = [ \
'detector_5_no_5_angle_2.jpg', \
'detector_7_no_5_angle_2.jpg'
]
__TrainImageDir__ = '/images/AfterPreprocessing/'

#preprocessing
pp__image_dir = '/images/original/'
pp__result_dir = __TrainImageDir__

#deeplearning
dp__image_dir = '/images/deepLearning/'
dp__result_filename = 'deepLearningResult.txt'
dp__accuracy_filename = 'deepLearningAccuracy.txt'

#curvature
cu__image_dir = '/images/curvature/'
cu__result_filename = 'curvatureResult.txt'
