function data = HoughLinearity()
% this function test linearity performace of hough transform
    DETECTOR_NUM=7; % number of detectors
    NO_NUM = 6; % detctor 6 has only five experiments
    ANGLE_NUM =3; % number of angles in taking images

    index = 0;
    manaul_count = [1 27 40 79 122 160 1 18 28 42 121 223 0 ...
        11 24 46 142 173 3 19 23 76 191	197 0 15 24 45 91 152 ...
        0 16 27	34 88 0	9 12 69	104 123];
    data = manaul_count';
    for det = 1:DETECTOR_NUM
        if(det == 6)
            num_n = NO_NUM-1;
        else
            num_n = NO_NUM;
        end
        for n = 0:(num_n-1)
            index = index+1;
            for angle = 1:ANGLE_NUM  
                filename = ['detector_' num2str(det) '_no_' num2str(n) '_angle_' num2str(angle) '.jpg'];
                refname = ['detector_' num2str(det) '_no_' num2str(n) '_background.jpg'];
                disp(['Processing ' filename ' ...']);
                system(['sudo python PreForMATLAB.py ' filename ' ' refname]);
                img = imread('pre_image.jpg');
                temp(angle)=HoughCircleDetector(img, 0);
            end
            data(index,2) = mean(temp);
            data(index,3) = std(temp);
        end
    end
    
end

