
clear all
clc
close all
addpath gaussianFunctions/;
addpath loopFunctions/;
addpath costFunctions/;
addpath trainFunctions/;
addpath evaluateFunctions/;
addpath plotting/;
addpath toolbox/;
addpath toolbox/minFunc/;
addpath toolbox/pwmetric/;
%plot_tsne
% BEGIN primary configurable parameters.
% - dataset is the image set we're using (CIFAR-10)
% - word set is the name of the folder within word_data
% containing word vectors (see README for details).
fields = {{'dataset',        'cifar10_small'};
          {'wordset',        'acl'};
          {'lambda_penalty',    1E-4};
};
% END primary configurable parameters.

% Load existing model parameters, if they exist
for i = 1:length(fields)
    if exist('fullParams','var') && isfield(fullParams,fields{i}{1})
        disp(['Using the previously defined parameter ' fields{i}{1}])
    else
        fullParams.(fields{i}{1}) = fields{i}{2};
    end
end

loadData; % Comment out if you've already loaded data.

disp('Training mapping function');
% Train mapping function
trainParams.imageDataset = fullParams.dataset;
trainParams.lambda_penalty = fullParams.lambda_penalty;


% for i =1: length(nonZeroCategories)
%     cat_id = nonZeroCategories(i);
%     [theta{i}, trainParams ] = trainMapping(X, Y, cat_id, trainParams, wordTable);
% 
% end
load('./gauss_cifar10_acl_cat_truck/theta.mat');
seen_label_names = label_names(nonZeroCategories);
mapDoEvaluate(X, Y, nonZeroCategories, label_names, seen_label_names, wordTable, theta, trainParams,outputPath, true);
save(sprintf('%s/theta.mat', outputPath), 'theta', 'trainParams');
    % Get train accuracy


disp('Training seen softmax features');
mappedCategories = zeros(1, numCategories);
% % mappedCategories = 1     2     3     0     4     5     6     7     8     0
mappedCategories(nonZeroCategories) = 1:numCategories-length(zeroCategories);
trainParamsSeen.nonZeroShotCategories = nonZeroCategories;
% %mappedCategories(Y) maps category labels from disjoint nonzeroshot category lables
% %like 1,2,3,5,6,7,8,9 to continuout values like 1,2,3,4,5,6,7,8
[thetaSeen, trainParamsSeen] = nonZeroShotTrain(X, mappedCategories(Y), trainParamsSeen, label_names(nonZeroCategories));
save(sprintf('%s/trainParams.mat', outputPath),  'trainParams');
save(sprintf('%s/thetaSeenSoftmax.mat', outputPath), 'thetaSeen', 'trainParamsSeen');

% % Get train accuracy
softmaxDoEvaluate( X, Y, label_names, thetaSeen, trainParamsSeen, true );
 
disp('Training unseen softmax features');
trainParamsUnseen.zeroShotCategories = zeroCategories;
trainParamsUnseen.imageDataset = fullParams.dataset;
trainParamsUnseen.wordDataset = fullParams.wordset;
[thetaUnseen, trainParamsUnseen] = zeroShotTrain(trainParamsUnseen);
save(sprintf('%s/thetaUnseenSoftmax.mat', outputPath), 'thetaUnseen', 'trainParamsUnseen');
% 
% % Train Gaussian classifier %mu 10x50, sigma: 10x1
D =trainParams.outputSize;
mu = zeros(numCategories, D );
sigma = zeros(numCategories,1);
disp('Training Gaussian classifier using Mixture of Gaussians');
priors = zeros(numCategories,1);
k = 1;
mapped = cell(length(nonZeroCategories),1);
for i =1:numCategories
    if ismember(i,nonZeroCategories)
        mapped{k} = mapDoMap(X, theta{k}, trainParams);
        [mu(i,:), sigma(i), priors(i)] = trainGaussianDiscriminant(mapped{k}, Y, i, numCategories, wordTable(:,i));
        k = k+1;
    else
        mu(i,:) = wordTable(:,i)';
        priors(i) = sum(Y == i)/ length(Y);
        
    end
end

save(sprintf('%s/mappedTrainImages.mat', outputPath), 'mapped');

[~,numTraining] = size(X);
sorted_train_Logprob = zeros(length(nonZeroCategories), numTraining );
mappedTestImages = cell(length(nonZeroCategories),1);
for i = 1: length(nonZeroCategories)
    cat_id = nonZeroCategories(i);
    sorted_train_Logprob(i,:) = predictGaussianDiscriminant(mapped{i}, mu(cat_id,:), sigma(cat_id), priors(cat_id));
    mappedTestImages{i} = mapDoMap(testX, theta{i}, trainParams);

end
save(sprintf('%s/mappedTestImages.mat', outputPath), 'mappedTestImages');

% % Test
%mappedTestImages = mapDoMap(testX, theta, trainParams);
sorted_train_Logprob = sort(sum(sorted_train_Logprob));
resolution = 11;
gSeenAccuracies = zeros(1, resolution);
gUnseenAccuracies = zeros(1, resolution);
gAccuracies = zeros(1, resolution);
numPerIteration = floor(length(sorted_train_Logprob) / (resolution-1));

for i = 1: length(nonZeroCategories)
    cat_id = nonZeroCategories(i);
    logprobabilities(i,:) = predictGaussianDiscriminant(mappedTestImages{i}, mu(cat_id,:), sigma(cat_id), priors(cat_id));
end
save(sprintf('%s/logprobabilities.mat', outputPath), 'logprobabilities');

%logprobabilities = sum(logprobabilities);

% sortedLogprobabilities (ascending order)= a   b   c   d   e   f   g   h
% if numPerIteration 3
%cutoffs = [a , d, g]
cutoffs = [ arrayfun(@(x) sorted_train_Logprob((x-1)*numPerIteration+1), 1:resolution-1) sorted_train_Logprob(end) ];
for i = 1:resolution
    cutoff = cutoffs(i);
    % Test Gaussian classifier
    fprintf('With cutoff %f:\n', cutoff);
    results = mapGaussianThresholdDoEvaluate( testX, mappedTestImages, testY, zeroCategories, label_names, wordTable, ...
        theta, trainParams, thetaSeen, trainParamsSeen, thetaUnseen, trainParamsUnseen, logprobabilities, cutoff, true);

    gSeenAccuracies(i) = results.seenAccuracy;
    gUnseenAccuracies(i) = results.unseenAccuracy;
    gAccuracies(i) = results.accuracy;
end
 gSeenAccuracies = fliplr(gSeenAccuracies);
 gUnseenAccuracies = fliplr(gUnseenAccuracies);
 gAccuracies = fliplr(gAccuracies);
% 
 plot_Gaussian_model
% plot_randomConfusionWords_6
