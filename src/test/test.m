%This file is only for tesing purpose, should not be included for final project
%the following code are tring to show histgram of pixels after edge detection
close all;
for det = 1:7
    if(det~=6)
        num_n = 5;
    else
        num_n = 4;
    end
    for n = 0:num_n
        for ang = 1:3
            filename = ['detector_' num2str(det) '_no_' num2str(n) '_angle_' num2str(ang) '.jpg'];
            img = rgb2gray(imread(filename));
            [v,h] = size(img);
            img_edge = edge(img,'canny',0.2,'both',5);
            index = find(img_edge == 1);
            y = mod(index, v);
            x = index/v;
            subplot(2,2,1);imshow(img);axis on;title(filename);
            subplot(2,2,4);imshow(img_edge);axis on;title('edges');
            subplot(2,2,2);hist(x,100);title('X');axis([1,h,1,v]);
            subplot(2,2,3);hist(y,100);title('Y');axis([1,v,1,h]);camroll(90);
            pause(0.5);
        end
    end
end
