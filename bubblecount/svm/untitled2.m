for det = 1:DETECTOR_NUM
        if(det == 6)
            num_n = NO_NUM-1;
        else
            num_n = NO_NUM;
        end
        for n = 0:(num_n-1)
            index = index+1;
            temp = zeros(ANGLE_NUM,1);
            for angle = 1:ANGLE_NUM  
                testfilename = ['../../images/detector_' num2str(det) '_no_' num2str(n) '_angle_' num2str(angle) '.jpg'];
                disp(['Processing ' testfilename ' ...']);
                clear testInstances;
                clear testingLabels;
                testInstances = testDataSetGeneration(testfilename, instanceSize, step);
                [testingLabels, score] = predict(SVMModel,testInstances');
                temp(angle) = length(find(testingLabels>0));
                
                label_index = find(testingLabels<=0);
                resultShow = imread(testfilename);
                pr = 1:step:(size(resultShow,1)-instanceSize+1);
                pc = 1:step:(size(resultShow,2)-instanceSize+1);
                lr = length(pr);
                lc = length(pc);
                for i=1:length(label_index)
                    r = ceil(label_index(i)/lc);
                    c = label_index(i)-(r-1)*lc;
                    resultShow(pr(r):(pr(r)+instanceSize-1),pc(c):(pc(c)+instanceSize-1),:) = 0;
                end
                imwrite(resultShow,['../../images/svm/' 'detector_' num2str(det) '_no_' num2str(n) '_angle_' num2str(angle) '.jpg'],'jpg'); 
            end
            data(index,2) = mean(temp);
            data(index,3) = std(temp);
        end
end
save('data.txt','data','-ascii');