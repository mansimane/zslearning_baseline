function [guessedLabels] = feedforwardDiscriminant(thetaMapping, thetaSoftmaxSeen, thetaSoftmaxUnseen, trainParams, trainParamsSeen, trainParamsUnseen, logprobabilities, images, mappedTestImages, maxLogprobability, zeroCategoryTypes, nonzeroCategoryTypes, wordTable)

addpath toolbox/pwmetric;
sum_logprobabilities = sum(logprobabilities);
% Forward Propagation
%mappedImages = mapDoMap(images, thetaMapping, trainParams);

unseenIndices = sum_logprobabilities < maxLogprobability;

seenIndices = ~unseenIndices;
% Seen label classifier
Ws = stack2param(thetaSoftmaxSeen, trainParamsSeen.decodeInfo);
pred = exp(Ws{1}*images(:, seenIndices)); % k by n matrix with all calcs needed
pred = bsxfun(@rdivide,pred,sum(pred));
[~, gind] = max(pred);
guessedLabels(seenIndices) = nonzeroCategoryTypes(gind);

% This is the unseen label classifier
unseenWordTable = wordTable(:, zeroCategoryTypes);
[~,closest_seen_idx] = min(logprobabilities(:,unseenIndices));
D = size(wordTable,1);
unseen_idx = find(unseenIndices==1);
closest_mapping = zeros(D,length(unseen_idx));
for i = 1:length(closest_seen_idx)
    closest_mapping(:,i) = mappedTestImages{closest_seen_idx(i)}(:,unseen_idx(i));
end

tDist = slmetric_pw(unseenWordTable, closest_mapping, 'eucdist');
[~, tGuessedCategories ] = min(tDist);
guessedLabels(unseenIndices) = zeroCategoryTypes(tGuessedCategories);
% find class which mapas given image closest to it e.g.
% Wu = stack2param(thetaSoftmaxUnseen, trainParamsUnseen.decodeInfo);
% pred = exp(Wu{1}*mappedImages(:, unseenIndices)); % k by n matrix with all calcs needed
% pred = bsxfun(@rdivide,pred,sum(pred));
% [~, gind] = max(pred);
% guessedLabels(unseenIndices) = zeroCategoryTypes(gind);

end
