function instancesLowDimension = dimensionReduction(instances, dimensions)
% Dimension Reduction With PCA Method

disp('Performing dimension reduction for instances');
[U,S,V] = svd(instances*instances'/size(instances,2));
W = U*S^(-1/2)*V';
instancesLowDimension = W(1:dimensions,:)*instances;
