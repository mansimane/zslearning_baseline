function [ guessedCategories, results ] = evaluateGaussianBayesian(thetaSeenSoftmax, thetaUnseenSoftmax, ...
    thetaMapping, seenSmTrainParams, unseenSmTrainParams, mapTrainParams, images, ...
    categories, cutoffs, zeroCategoryTypes, nonZeroCategoryTypes, categoryNames, wordVectors, doPrint)

addpath toolbox;

numImages = size(images, 2);
numCategories = length(zeroCategoryTypes) + length(nonZeroCategoryTypes);
Ws = stack2param(thetaSeenSoftmax, seenSmTrainParams.decodeInfo);
Wu = stack2param(thetaUnseenSoftmax, unseenSmTrainParams.decodeInfo);

mappedImages = mapDoMap(images, thetaMapping, mapTrainParams);

% This is the seen label classifier
probSeen = exp(Ws{1}*images); % k by n matrix with all calcs needed
probSeen = bsxfun(@rdivide,probSeen,sum(probSeen));
probSeenFull = zeros(numCategories, numImages);
probSeenFull(nonZeroCategoryTypes, :) = probSeen;

% This is the unseen label classifier
probUnseen = exp(Wu{1}*mappedImages); % k by n matrix with all calcs needed
probUnseen = bsxfun(@rdivide,probUnseen,sum(probUnseen));
probUnseenFull = zeros(numCategories, numImages);
probUnseenFull(zeroCategoryTypes, :) = probUnseen;

% Treat everything as seen first, then filter out cases
% where things fall outside cutoff circles
probs = ones(size(categories));
for c_i = 1:length(nonZeroCategoryTypes)
    currentCategory = nonZeroCategoryTypes(c_i);
    centerVector = wordVectors(:, currentCategory);
    dists = slmetric_pw(centerVector, mappedImages, 'eucdist');
    probs(dists < cutoffs(currentCategory)) = 0; % falls in circle; is not unseen    
end
%probs = 0 Seen, probs = 1 Unseen
finalProbs = bsxfun(@times, probSeenFull, 1 - probs') + bsxfun(@times, probUnseenFull, probs');
[~, guessedCategories ] = max(finalProbs);

% Calculate scores
confusion = zeros(numCategories, numCategories);
for actual = 1:numCategories
    guessesForCategory = guessedCategories(categories == actual);
    for guessed = 1:numCategories
        confusion(actual, guessed) = sum(guessesForCategory == guessed);
    end
end

truePos = diag(confusion); % true positives, column vector
results.accuracy = sum(truePos) / numImages;
numUnseen = sum(arrayfun(@(x) nnz(categories == x), zeroCategoryTypes));
results.unseenAccuracy = sum(truePos(zeroCategoryTypes)) / numUnseen;
results.seenAccuracy = (sum(truePos) - sum(truePos(zeroCategoryTypes))) / (numImages - numUnseen);
t = truePos ./ sum(confusion, 2);
results.avgPrecision = mean(t(isfinite(t), :));
t = truePos' ./ sum(confusion, 1);
results.avgRecall = mean(t(:, isfinite(t)));
results.confusion = confusion;

if doPrint == true
    disp(['Accuracy: ' num2str(results.accuracy)]);
    disp(['Seen Accuracy: ' num2str(results.seenAccuracy)]);
    disp(['Unseen Accuracy: ' num2str(results.unseenAccuracy)]);
    disp(['Averaged precision: ' num2str(results.avgPrecision)]);
    disp(['Averaged recall: ' num2str(results.avgRecall)]);
    displayConfusionMatrix(confusion, categoryNames);
end

end

