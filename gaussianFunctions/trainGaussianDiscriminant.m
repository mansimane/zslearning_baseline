%p(y|x,X_T) = sum_v p(y|v,X_t)*p(v|x,X_T)
%p(y=1|v,x,X_T) = p(x|y=i,mu,Sigma)*p(y=i)/sum_{k=1}^{|V|}p(x|y=k,mu,Sigma)*pi(y=k)

function [mu,sigma_elem,prior] = trainGaussianDiscriminant(projectedImageFeatures, labels, cat_id, numLabels, wordVectors)

[dim,numTraining] = size(projectedImageFeatures);
%sigma = zeros(numLabels, dim, dim);
%mu = zeros(numLabels, dim);
mu = wordVectors';

%priors = zeros(numLabels, 1);

labelImageFeatures = projectedImageFeatures(:, labels == cat_id);
prior = size(labelImageFeatures, 2) / numTraining;
labelMu = mu';
% divinding sigma by only number of images belonging to that class
% Earlier they were diviging by total N
sigma_elem = sum(sum(bsxfun(@minus, labelImageFeatures, labelMu).^2))/(size(labelImageFeatures,2)*dim);
%    sigma(i,:,:) = diag(repmat(sum(sum(bsxfun(@minus, labelImageFeatures, labelMu).^2))/(numTraining*dim), dim, 1));


end
