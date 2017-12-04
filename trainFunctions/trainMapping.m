function [theta, trainParams] = trainMapping(X, Y, Xvalidate, Yvalidate, cat_id, trainParams, wordTable, outputPath, theta_idx)

addpath toolbox/;
addpath toolbox/minFunc/;
addpath toolbox/pwmetric/;
addpath costFunctions/;

%% Model Parameters
fields = {{'wordDataset',         'acl'};            % type of embedding dataset to use ('turian.200', 'acl')
          {'lambda',              1E-4};   % regularization parameter
          {'lambda_penalty',      0.01};   % regularization parameter
          {'numReplicate',        0};     % one-shot replication
          {'dropoutFraction',     1};    % drop-out fraction
          {'costFunction',        @mapTrainingCostOneLayer}; % training cost function
          {'trainFunction',       @trainLBFGS}; % training function to use
          {'hiddenSize',          200};
          {'maxIter',             150};    % maximum number of minFunc iterations on a batch
          {'maxPass',             1};      % maximum number of passes through training data
          {'disableAutoencoder',  true};   % whether to disable autoencoder
          {'maxAutoencIter',      50};     % maximum number of minFunc iterations on a batch
          
          % options
          {'batchFilePrefix',     'default_batch'};  % use this to choose different batch sets (common values: default_batch or mini_batch)
          {'zeroFilePrefix',      'zeroshot_batch'}; % batch for zero shot images
          {'fixRandom',           false};  % whether to fix the random number generator
          {'enableGradientCheck', false};  % whether to enable gradient check
          {'preTrain',            true};   % whether to train on non-zero-shot first
          {'reloadData',          true};   % whether to reload data when this script is called (disable for batch jobs)
          
          % Old parameters, just keep for compatibility
          {'saveEvery',           5};      % number of passes after which we need to do intermediate saves
          {'oneShotMult',         1.0};    % multiplier for one-shot multiplier
          {'autoencMultStart',    0.01};   % starting value for autoenc mult
          {'sparsityParam',       0.035};  % desired average activation of the hidden units.
          {'beta',                5};      % weight of sparsity penalty term
          {'epochs',              100};      % Number of epochs to train for
          {'batch_size',          256};    % Batchsize for gradient calcultation
          {'lr',                  0.001};    % Batchsize for gradient calcultation
};

% Load existing model parameters, if they exist
for i = 1:length(fields)
    if exist('trainParams','var') && isfield(trainParams,fields{i}{1})
        disp(['Using the previously defined parameter ' fields{i}{1}])
    else
        trainParams.(fields{i}{1}) = fields{i}{2};
    end
end

trainParams.f = @tanh;             % function to use in the neural network activations
trainParams.f_prime = @tanh_prime; % derivative of f
trainParams.doEvaluate = false;
trainParams.testFilePrefix = 'zeroshot_test_batch';
trainParams.autoencMult = trainParams.autoencMultStart;

trainParams.imageColumnSize = size(X, 1);

trainParams.costFunction = @mapTrainingCost;

% Initialize actual weights
disp('Initializing parameters');
trainParams.inputSize = trainParams.imageColumnSize;
trainParams.outputSize = size(wordTable, 1);
[ theta, trainParams.decodeInfo ] = initializeParameters(trainParams);

globalStart = tic;
% dataToUse.imgs = X;
% dataToUse.categories = Y;
dataToUse.wordTable = wordTable;
dataToUse.cat_id = cat_id;
batch_size = trainParams.batch_size;
no_of_samples = size(X,2);
train_iter = no_of_samples/ batch_size ;
indices = randperm(no_of_samples);
cost_array = zeros(1,trainParams.epochs);
cost_val_array = zeros(1,trainParams.epochs);

data_val.imgs = Xvalidate;
data_val.categories = Yvalidate; 
data_val.wordTable = wordTable;
data_val.cat_id = cat_id;

for i = 1: trainParams.epochs
    indices = randperm(no_of_samples);
    for j = 1:batch_size:no_of_samples-batch_size
         start_idx = j;
         end_idx = j+batch_size;
%         
%         if start_idx> end_idx
%             indices = randperm(no_of_samples);
%             continue
%         end
        idx = indices(start_idx:end_idx);
        dataToUse.imgs = X(:,idx);
        dataToUse.categories = Y(idx);

        [cost, grad] = mapTrainingCost( theta, dataToUse, trainParams );
        [theta] = updateParam(theta,grad, trainParams);
        
        cost_val = calcCost(theta, data_val, trainParams);
    end
    
    fprintf('Epoch = %d, train loss = %d,Val loss = %d\n', i, cost, cost_val);
    cost_array(i) = cost;
    cost_val_array(i) = cost_val;
end
%theta = trainParams.trainFunction(trainParams, dataToUse, theta);
figure;
plot(cost_array);
hold on;
plot(cost_val_array);
xlabel('epochs');
ylabel('MSE');
title('Loss vs Epochs');
legend('Train loss','Val loss');
file_name = [outputPath '/training_error_map_' num2str(theta_idx) '.jpg'];
Image = getframe(gcf);
imwrite(Image.cdata, file_name);

gtime = toc(globalStart);
fprintf('Total time: %f s\n', gtime);

end
