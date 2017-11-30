% outputs the probability that each image feature is seen before
function [logprobability] = predictGaussianDiscriminant(projectedImageFeatures, mu, sigma_elem, priors)

dim = size(mu, 1);
numTraining = size(projectedImageFeatures, 2);

probability = zeros(1, numTraining);
temp = bsxfun(@minus, projectedImageFeatures, mu');
logprobability = -0.5*(sum(1/sigma_elem*(temp.^2), 1) + dim*log(2*pi) + dim*log(sigma_elem));
logprobability = log(priors)+logprobability;

end
