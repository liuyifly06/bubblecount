% Convert a test images into tesing instances for SVM
function testInstances = testDataSetGeneration(filename, instanceSize, step)

disp(['Generating test data for image ' filename]);

test = imread(filename);
[m, n, c] = size(test);

index = 0;

totalInstancesX = length(1:step:(n-instanceSize+1));
totalInstancesY = length(1:step:(m-instanceSize+1));

for i = 1:step:(m-instanceSize+1)
    for j = 1:step:(n-instanceSize+1)
        index = index + 1;
        testInstances(:, index) = ...
            double(reshape(test(i:(i+instanceSize-1), ...
            j:(j+instanceSize-1),:), instanceSize^2 * c, 1));
    end
end
