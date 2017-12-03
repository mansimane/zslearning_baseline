% t-SNE visualize_script 
% Don't Run from Main
%WARNING: Needs lots of RAM !

% uncomment depending on what you want to map:
% training data (no zero shot classes) or test data (includes zero shot classes)
%  load('mappedTrainData.mat');
%load('./gauss_cifar10_acl_cat_truck/mappedTestImages.mat');
load('./gauss_cifar10_small_acl_cat_truck_0.0001/mappedTestImages.mat');
load('./word_data/acl/cifar10/wordTable.mat')
% load label names
load('image_data/images/cifar10/meta.mat');

% load word table
load('word_data/acl/cifar10/wordTable.mat');
load('image_data/features/cifar10_small/test.mat');
Y = testY+1;
label_names = [{'Not\_Label'}; label_names];
numImages = size(mappedTestImages{2}, 2);
num_seen_cats = 8;
% for i=1:num_seen_cats
%     disp(i);
%     t = tsne([mappedTestImages{2} wordTable]');
%     [C,t,l]= pca([mappedTestImages{2} wordTable]','NumComponents',2);
% 
%     mappedX_t = t(1:numImages, :);
%     mappedWordTable_t = t(numImages+1:end, :);
% 
%     do the visualization
%     visualize(mappedX_t, Y, mappedWordTable_t, label_names);
%     figure;
%     Y_bin = (Y==i);
%     Y_bin = Y_bin* i;
%     Y_bin = Y_bin + 1;
%     hold on;
%     gscatter(mappedX_t(:,1), mappedX_t(:,2), label_names(Y_bin), [], '+o*.xsd^v><', 8);
%     hold on;
% end
% hold on;
% 
% scatter(mappedWordTable_t(:,1), mappedWordTable_t(:,2), 200, 'd', 'k', 'filled');
% for i = 1:length(label_names)-1
%     text(mappedWordTable_t(i,1),mappedWordTable_t(i,2),label_names{i+1},'BackgroundColor',[.7 .9 .7]);        
% end
% axis off;
% hold off;

%% mapping everything in same space
    %t = tsne([mappedTestImages{2} wordTable]');
num_seen_cats = 8;

% [C,t,l]= pca([mappedTestImages{1} mappedTestImages{2} mappedTestImages{3} ...
%     mappedTestImages{4} mappedTestImages{5} mappedTestImages{6} ...
%     mappedTestImages{7} mappedTestImages{8} wordTable]','NumComponents',2);
t = tsne([mappedTestImages{1} mappedTestImages{2} mappedTestImages{3} mappedTestImages{4} mappedTestImages{5} mappedTestImages{6}  mappedTestImages{7} mappedTestImages{8} wordTable]');
    
mappedWordTable_t = t(numImages*num_seen_cats+1:end, :);
sym_array = '+o*.xsd^v><';
for i=1:num_seen_cats
    mappedX_t = t((i-1)*numImages +1 : (i-1)*numImages + numImages,:);
    %idx = find(Y==i);
    Y_bin = (Y==i);
    Y_bin = (Y_bin* i) + 1;
    hold on;
    syms = [sym_array(1), sym_array(max(Y_bin))];
    gscatter(mappedX_t(:,1), mappedX_t(:,2), label_names(Y_bin), [], syms, 2);
end
scatter(mappedWordTable_t(:,1), mappedWordTable_t(:,2), 200, 'd', 'k', 'filled');
for i = 1:length(label_names)-1
    text(mappedWordTable_t(i,1),mappedWordTable_t(i,2),label_names{i+1},'BackgroundColor',[.7 .9 .7]);        
end
axis off;
hold off;

    % do the visualization
    %visualize(mappedX_t, Y, mappedWordTable_t, label_names);
    %figure;
%     Y_bin = (Y==i);
%     Y_bin = Y_bin* i;
%     Y_bin = Y_bin + 1;
%     hold on;
%     gscatter(mappedX_t(:,1), mappedX_t(:,2), label_names(Y_bin), [], '+o*.xsd^v><', 8);
%     hold on;
% 
% hold on;
% 
