function image_show(filename, dir, threshold)
    original_img = imread(['./AfterPreprocessing/' filename]);
    labeled_img =imread([dir filename]);
    [m_o,n_o] = size(original_img(:,:,1));
    [m_l,n_l] = size(labeled_img);
    index = find(labeled_img >= threshold);
    pos = [floor(index/m_l), mod(index,m_l)];
    pos(:,1) = pos(:,1)*n_o/n_l;
    pos(:,2) = pos(:,2)*m_o/m_l;
    image(original_img);hold on;
    scatter(pos(:,1),pos(:,2),'.','r');
end