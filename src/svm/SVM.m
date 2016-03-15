% Analysis image with SVM Method
instanceSize = 20;
step = 10;
edge = 4;
scale = 60;

dimensionReduction = false;
ReducedDimentions = 20;

testfilename = '../../images/detector_7_no_5_angle_2.jpg'; 
[trainInstances, trainLabels] = trainDataSetGeneration(instanceSize, step, edge, scale);
testInstances = testDataSetGeneration(testfilename, instanceSize, step);

if(dimensionReduction)
    disp('Training data dimension reduction ...');
    trainLowDimension = dimensionReduction(trainInstances, ReducedDimentions);
    disp('Testing data dimension reduction ...');
    testLowDimension = dimensionReduction(testInstances, ReducedDimentions);
end

disp('SVM training ... ');

if(dimensionReduction)
    SVMModel = fitcsvm(trainLowDimension',trainLabels,'KernelFunction','rbf','KernelScale','auto');
    %'linear','gaussian', 'rbf','polynomial'
%    CVSVMModel = crossval(SVMModel);
%    L = kfoldLoss(CVSVMModel);
    disp('Labeling testing instances ... ');
    [testingLabels, score] = predict(SVMModel,testLowDimension');    
else
    SVMModel = fitcsvm(trainInstances',trainLabels,'KernelFunction','rbf','KernelScale','auto','Prior',[0.98,0.02]);
    %'linear','gaussian', 'rbf','polynomial'
 %   CVSVMModel = crossval(SVMModel);
 %   L = kfoldLoss(CVSVMModel);
    disp('Labeling testing instances ... ');
    [testingLabels, score] = predict(SVMModel,testInstances');
end


index = find(testingLabels>0);
resultShow = imread(testfilename);

pr = 1:step:(size(resultShow,1)-instanceSize+1);
pc = 1:step:(size(resultShow,2)-instanceSize+1);
lr = length(pr);
lc = length(pc);

for i=1:length(index)
    r = ceil(index(i)/lc);
    c = index(i)-(r-1)*lc;
    resultShow(pr(r):(pr(r)+instanceSize-1),pc(c):(pc(c)+instanceSize-1),1) = resultShow(pr(r):(pr(r)+instanceSize-1),pc(c):(pc(c)+instanceSize-1),1)+255;
end

figure('name','Pool Recognition','NumberTitle','off')
subplot(1,2,1);imshow(imread(testfilename));title('original Image');
subplot(1,2,2);imshow(resultShow);title('Red region is the recognized bubbles(SVM method)');
