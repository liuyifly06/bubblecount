function data = HoughLinearity()
% this function test linearity performace of hough transform
    DETECTOR_NUM=7; % number of detectors
    NO_NUM = 6; % detctor 6 has only five experiments
    ANGLE_NUM =3; % number of angles in taking images

    index = 0;
    for det = 1:DETECTOR_NUM
        if(det == 6)
            num_n = NO_NUM-1;
        else
            num_n = NO_NUM;
        end
        for n = 0:(num_n-1)            
            for angle = 1:ANGLE_NUM
                index = index+1;
                filename = ['detector_' num2str(det) '_no_' num2str(n) '_angle_' num2str(angle) '.jpg'];
                %refname = ['detector_' num2str(det) '_no_' num2str(n) '_background.jpg'];
                disp(['Processing ' filename ' ...']);
                img = rgb2gray(imread(['../../images/AfterPreprocessing/' filename]));
                data(index)= HoughCircleDetector(img, 0);
            end
        end
    end
    save data
end

