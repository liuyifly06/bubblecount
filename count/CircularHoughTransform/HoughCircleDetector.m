function NumberOfCircles = HoughCircleDetector(img, plot_show)
%Caculate number of circles by using hough transform

[accumulation_array, centers, radii] = HoughCircleTransform(img, [2 25]);

% Visualize the accumulation array
% figure(1); imagesc(accumulation_array); axis image;
% title('Accumulation Array from Circular Hough Transform');

if (plot_show==1)
    % plot the recognized circles
    figure(2); subplot(1,3,3);h=imagesc(img'); colormap('gray'); axis image;
    hold on; plot(centers(:,2), centers(:,1), 'r+');
    for k = 1 : size(centers, 1),
        DrawCircle(centers(k,2), centers(k,1), radii(k), 32, 'r-');
    end
    hold off;
    title(['Circles Detected ', ...
    'with Hough Transform']);
end

% 3D view of the local maxima
% figure(3); surf(accumulation_array, 'EdgeColor', 'none'); axis ij;
% title('3-D View of the Accumulation Array');

%Number of circles detected
NumberOfCircles = length(centers); 
end

