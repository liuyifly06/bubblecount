% Generating training instances for SVM
function [trainInstances, trainLabels] = trainDataSetGeneration(instanceSize, step, edge, scale)

instances_show = 0;

disp(['Generating training instances ... instanceSize: ' ...
    num2str(instanceSize) ' step: ' num2str(step)]);

train_file = '../../images/detector_5_no_5_angle_2.jpg';
train = imread(train_file);
load('positiveInstances.mat');
[m,n,c] = size(train);

positiveLabels = positiveInstances.objectBoundingBoxes;
x_c = positiveLabels(:,1) + ceil(positiveLabels(:,3)/2);
y_c = positiveLabels(:,2) + ceil(positiveLabels(:,4)/2);
a = ceil(positiveLabels(:,3)/2)-edge;
b = ceil(positiveLabels(:,4)/2)-edge;

numPositive = 0;
numNegative = 0;

totalInstancesX = length(max(1,floor(instanceSize/2)) : step ...
    : (m - ceil(instanceSize/2)));
totalInstancesY = length( max(1,floor(instanceSize/2)) ...
    : step : (n - ceil(instanceSize/2)));

progressRatePrevious = 0;

Instances = zeros(instanceSize^2 * c, totalInstancesX * totalInstancesY);
Labels = zeros(1,totalInstancesX * totalInstancesY);
tic;
ind = 0;
for i = max(1,floor(instanceSize/2)) : step : (m - ceil(instanceSize/2))    
    for j = max(1,floor(instanceSize/2)) : step : (n - ceil(instanceSize/2))
        
        currentY = i;
        currentX = j;
        criterion = min(((currentX - x_c)./a).^2 + ((currentY - y_c)./b).^2);
        
        boundaryT = currentY - floor(instanceSize/2) + 1;
        boundaryD = currentY + floor(instanceSize/2);
        boundaryL = currentX - floor(instanceSize/2) + 1;
        boundaryR = currentX + floor(instanceSize/2);
        
        ind = ind + 1;
        
        temp = double(train(boundaryT : boundaryD, boundaryL : boundaryR, 1 : c));
        Instances(:, ind) = reshape(temp,instanceSize^2 * c,1);
        
        
        if (criterion <= 1)
            
            numPositive = numPositive + 1;
            Labels(ind) = 1;
            
        else
            numNegative = numNegative + 1;
            Labels(ind) = -1;
            
        end     
    end
    
    elapse_time = toc;
    progressRate = ind / totalInstancesX / totalInstancesY * 100;
    remain_time = toc/progressRate*100 - elapse_time;
    progressRate = floor(progressRate/10);
    if progressRate > progressRatePrevious
        disp([num2str(progressRate*10) ...
            '% train instances created, remaining time: ' ...
            num2str(ceil(remain_time)) ' s']);
    end
    progressRatePrevious = progressRate;
end

instancesPositive = Instances(:,Labels==1);
instancesNegative = Instances(:,Labels==-1);

numOfTrainPositiveInstances  = numPositive;
numOfTrainNegativeInstances  = numOfTrainPositiveInstances*scale;

indexPositive = 1:numOfTrainPositiveInstances;
indexNegative = ceil(rand(numOfTrainNegativeInstances,1)*size(instancesNegative,2));

instancesPositive = instancesPositive(:,indexPositive);
instancesNegative = instancesNegative(:,indexNegative);

trainInstances = [instancesPositive, instancesNegative];
trainLabels = [ones(1,numOfTrainPositiveInstances),-ones(1,numOfTrainNegativeInstances)];

imgPositive = zeros(floor(sqrt(numPositive))*10,floor(sqrt(numPositive))*10, c);
imgNegative = zeros(floor(sqrt(numPositive))*10,floor(sqrt(numPositive))*10, c);

if(instances_show == 1)
    for i = 1:floor(sqrt(numPositive))
        for j = 1:floor(sqrt(numPositive))
            index = floor(sqrt(numPositive))*(i-1) + j;
            imgPositive((instanceSize*(i-1)+1):instanceSize*i, ...
                (instanceSize*(j-1)+1):instanceSize*j, 1 : c) = ...
                reshape(instancesPositive(:,index), ...
                instanceSize, instanceSize, c);
            imgNegative((instanceSize*(i-1)+1):instanceSize*i, ...
                (instanceSize*(j-1)+1):instanceSize*j, 1 : c) = ...
                reshape(instancesNegative(:,index), ...
                instanceSize, instanceSize, c);
        end
    end
    figure('name','Training Instances');
    subplot(1,2,1);imshow(uint8(imgPositive));title('Positive Instances');
    subplot(1,2,2);imshow(uint8(imgNegative));title('Negative Instances');
end
