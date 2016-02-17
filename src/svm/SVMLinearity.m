% function for testing linearity of SVM method
function data = SVMLinearity()
    DETECTOR_NUM=7;
    NO_NUM = 6;
    ANGLE_NUM =3;

    index = 0;
    manaul_count = [1	27	40	79	122	160	1	18	28	42	121	223	0 ...
        11	24	46	142	173	3	19	23	76	191	197	0	15	24	45	91	152 ...
        0	16	27	34	88	0	9	12	69	104	123];
    data = manaul_count';
    
    
    instanceSize = 30;
    step = 10;
    edge = 5;
    scale = 60;
    
    [trainInstances, trainLabels] = trainDataSetGeneration(instanceSize, step, edge, scale);
    testInstances = testDataSetGeneration(testfilename, instanceSize, step);
    disp('SVM training ... ');
    SVMModel = fitcsvm(trainInstances',trainLabels,'KernelFunction','rbf','KernelScale','auto');
    
    temp = zeros(ANGLE_NUM,1);
    for det = 1:DETECTOR_NUM
        if(det == 6)
            num_n = NO_NUM-1;
        else
            num_n = NO_NUM;
        end
        for n = 0:(num_n-1)
            index = index+1;
            for angle = 1:ANGLE_NUM
                testfilename = ['detector_' num2str(det) '_no_' num2str(n) '_angle_' num2str(angle) '.jpg'];
                img = imread(testfilename);

                disp(['Labeling testing instances in ' testfilename]);
                [testingLabels, score] = predict(SVMModel,testInstances');
                index = find(testingLabels>0);
                temp(angle) = length(index);              
            end
            data(index,2) = mean(temp);
            data(index,3) = std(temp);
        end
    end
end
