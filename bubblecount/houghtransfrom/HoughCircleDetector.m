function NumberOfCircles = HoughCircleDetector(img, plot_show)
%Caculate number of circles by using hough transform

[accumulation_array, centers, radii] = HoughCircleTransform(img, [2 25]);

% Visualize the accumulation array
figure(1); imagesc(accumulation_array); axis image;
title('Voting Result from Circular Hough Transform');
aa = im2double(accumulation_array);
imwrite(sqrt(aa./max(aa(:))), '../../images/HoughTransform/vote.png')
if (plot_show==1)
    % plot the recognized circles
    figure(2); h = imagesc(img); colormap('gray'); axis image;
    hold on; plot(centers(:,1), centers(:,2), 'r+');
    for k = 1 : size(centers, 1),
        DrawCircle(centers(k,1), centers(k,2), radii(k), 32, 'r-');
    end
    hold off;
    title(['Circles Detected ', ...
    'with Hough Transform']);
end



% 3D view of the local maxima
% figure(3); surf(accumulation_array, 'EdgeColor', 'none'); axis ij;
% title('3-D View of the Accumulation Array');

%Number of circles detected
%NumberOfCircles = length(centers);
NumberOfCircles = sum(accumulation_array(:));
end

