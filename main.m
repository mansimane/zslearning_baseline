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

% BEGIN primary configurable parameters.
% - dataset is the image set we're using (CIFAR-10)
% - word set is the name of the folder within word_data
% containing word vectors (see README for details).
fields = {{'dataset',        'cifar10_small'};
          {'wordset',        'acl'};
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

% for i =1: length(nonZeroCategories)
%     cat_id = nonZeroCategories(i);
%     [theta{i}, trainParams ] = trainMapping(X, Y, cat_id, trainParams, wordTable);
% 
% end
load('./gauss_cifar10_acl_cat_truck/theta.mat');
seen_label_names = label_names(nonZeroCategories);
mapDoEvaluate(X, Y, nonZeroCategories, label_names, seen_label_names, wordTable, theta, trainParams, true);
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
[~,numTraining] = size(X);
sorted_train_Logprob = zeros(length(nonZeroCategories), numTraining );
mappedTestImages = cell(length(nonZeroCategories),1);
for i = 1: length(nonZeroCategories)
    cat_id = nonZeroCategories(i);
    sorted_train_Logprob(i,:) = predictGaussianDiscriminant(mapped{i}, mu(cat_id,:), sigma(cat_id), priors(cat_id));
    mappedTestImages{i} = mapDoMap(testX, theta{i}, trainParams);

end
    % 
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
 plot_randomConfusionWords_6
%% 

% disp('Training LoOP model'); fullParams.resolution = resolution;
% resolution = fullParams.resolution - 1; thresholds = 0:(1/resolution):1;
% lambdas = 1:13; knn = 20; loopSeenAccuracies = zeros(length(lambdas),
% length(thresholds)); loopUnseenAccuracies = zeros(length(lambdas),
% length(thresholds)); loopAccuracies = zeros(length(lambdas),
% length(thresholds)); nonZeroCategoryIdPerm =
% randperm(length(nonZeroCategories)); bestLambdas =
% repmat(lambdas(round(length(lambdas)/2)), 1, length(nonZeroCategories));
% mappedValidationImages = mapDoMap(Xvalidate, theta, trainParams);
% 
% for k = 1:length(nonZeroCategories)
%     changedCategory = nonZeroCategoryIdPerm(k); for i = 1:length(lambdas)
%         tempLambdas = bestLambdas; tempLambdas(changedCategory) =
%         lambdas(i); disp(tempLambdas); [ nplofAll, pdistAll ] =
%         trainOutlierPriors(mapped, Y, nonZeroCategories, numTrainPerCat,
%         knn, tempLambdas); probs = calcOutlierPriors(
%         mappedValidationImages, mapped, Y, numTrainPerCat,
%         nonZeroCategories, tempLambdas, knn, nplofAll, pdistAll ); for t
%         = 1:length(thresholds)
%             fprintf('Threshold %f: ', thresholds(t)); [~, results] =
%             anomalyDoEvaluate(thetaSeen, ...
%                 trainParamsSeen, thetaUnseen, trainParamsUnseen, probs,
%                 Xvalidate, mappedValidationImages, Yvalidate, ...
%                 thresholds(t), zeroCategories, nonZeroCategories,
%                 wordTable, false);
%             loopSeenAccuracies(i, t) = results.seenAccuracy;
%             loopUnseenAccuracies(i, t) = results.unseenAccuracy;
%             loopAccuracies(i, t) = results.accuracy; fprintf('seen
%             accuracy: %f, unseen accuracy: %f\n', results.seenAccuracy,
%             results.unseenAccuracy);
%         end
%     end [~, t] = max(sum(loopAccuracies,2)); bestLambdas(changedCategory)
%     = t;
% end disp('Best:'); disp(bestLambdas); % Do it again, with best lambdas
% loopSeenAccuracies = zeros(1, length(thresholds)); loopUnseenAccuracies =
% zeros(1, length(thresholds)); loopAccuracies = zeros(1,
% length(thresholds)); [ nplofAll, pdistAll ] = trainOutlierPriors(mapped,
% Y, nonZeroCategories, numTrainPerCat, knn, bestLambdas); probs =
% calcOutlierPriors( mappedTestImages, mapped, Y, numTrainPerCat,
% nonZeroCategories, bestLambdas, knn, nplofAll, pdistAll ); for t =
% 1:length(thresholds)
%     fprintf('Threshold %f: ', thresholds(t));
%             [~, results] = anomalyDoEvaluate(thetaSeen, ...
%                 trainParamsSeen, thetaUnseen, trainParamsUnseen, probs,
%                 testX, mappedTestImages, testY, ... thresholds(t),
%                 zeroCategories, nonZeroCategories, wordTable, false);
%     loopSeenAccuracies(t) = results.seenAccuracy; loopUnseenAccuracies(t)
%     = results.unseenAccuracy; loopAccuracies(t) = results.accuracy;
%     fprintf('accuracy: %f, seen accuracy: %f, unseen accuracy: %f\n',
%     results.accuracy, results.seenAccuracy, results.unseenAccuracy);
% end save(sprintf('%s/bestLambdas.mat', outputPath), 'bestLambdas');
% 
% disp('Run Bayesian pipeline for LoOP'); [~, bayesianResult] =
% evaluateLoopBayesian(thetaSeen, thetaUnseen, ...
%     theta, trainParamsSeen, trainParamsUnseen, trainParams, mapped, Y,
%     testX, ... testY, bestLambdas, knn, nplofAll, pdistAll,
%     numTrainPerCat, zeroCategories, nonZeroCategories, label_names,
%     true);
% 
% %%%%%%
% 
% cutoffs = generateGaussianCutoffs(thetaSeen, thetaUnseen, theta,
% trainParamsSeen, ...
%   trainParamsUnseen, trainParams, X, Y, wordTable, 0.05, 1,
%   zeroCategories, nonZeroCategories);
% 
% disp('Run Bayesian pipeline for Gaussian model'); thetaSeenSoftmax =
% thetaSeen; thetaUnseenSoftmax =  thetaUnseen; seenSmTrainParams =
% trainParamsSeen; unseenSmTrainParams = trainParamsUnseen; mapTrainParams
% = trainParams; thetaMapping = theta;
% 
% validX = Xvalidate; validY =Yvalidate; [ guessedCategories, results ] =
% evaluateGaussianBayesian(thetaSeenSoftmax, thetaUnseenSoftmax, ...
%     thetaMapping, seenSmTrainParams, unseenSmTrainParams, mapTrainParams,
%     validX, ... validY, cutoffs, zeroCategories, nonZeroCategories,
%     label_names, wordTable, true);
% 
% 
% 
% pdfSeenAccuracies = results.seenAccuracy; pdfUnseenAccuracies =
% results.unseenAccuracy; pdfAccuracies = results.accuracy;
% 
% %Save results. save(sprintf('%s/out_%s.mat', outputPath, zeroStr),
% 'gSeenAccuracies', 'gUnseenAccuracies', 'gAccuracies', ...
%     'loopSeenAccuracies', 'loopUnseenAccuracies', 'loopAccuracies',
%     'pdfSeenAccuracies', 'pdfUnseenAccuracies', ... 'pdfAccuracies',
%     'bayesianResult');

%Plot graphs
% plot_unseen_bar_3
%plot_modelComparisons_4 plot_randomConfusionWords_6
