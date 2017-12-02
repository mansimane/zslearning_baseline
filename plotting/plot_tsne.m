% t-SNE visualize_script 
%WARNING: Needs lots of RAM !

% uncomment depending on what you want to map:
% training data (no zero shot classes) or test data (includes zero shot classes)
%  load('mappedTrainData.mat');
load('.\gauss_cifar10_acl_cat_truck\mappedTestImages.mat');
load('word_data\acl\cifar10\wordTable.mat')
% load label names
load('image_data/images/cifar10/meta.mat');

% load word table
load('word_data/acl/cifar10/wordTable.mat');
load('image_data\features\cifar10\test.mat');
Y = testY+1;

numImages = size(mappedTestImages{2}, 2);
num_seen_cats = 8;
for i=1:num_seen_cats
    t = tsne([mappedTestImages{2} wordTable]');
    mappedX_t = t(1:numImages, :);
    mappedWordTable_t = t(numImages+1:end, :);

    % do the visualization
    %visualize(mappedX_t, Y, mappedWordTable_t, label_names);
    figure;
    Y_bin = (Y==i);
    Y_bin = Y_bin* i;
    gscatter(mappedX_t(:,1), mappedX_t(:,2), label_names(Y_bin), [], '+o', 2);
end