function InstancesToFile()
    load('positiveInstances.mat');
    [fileID,errmsg] = fopen('positiveInstances.dat','wt');
    disp(errmsg);
    % format :  image file name + number of regions + region detail
    for i = 1:length(positiveInstances);
        image_filename = positiveInstances(i).imageFilename;
        positive_region = positiveInstances(i).objectBoundingBoxes;
        [m,n] = size(positive_region);
        positive_region = reshape(positive_region', 1, m*n);
        % file name
        fprintf(fileID,'%s',image_filename((end-26):end));
        % number of regions
        fprintf(fileID,' %d', m);
        % region detail
        for j = 1 : (m*n)
            fprintf(fileID, ' %d', positive_region(1,j));
        end
        % go to next line
        fprintf(fileID,'\n');
    end
    fclose(fileID);
end