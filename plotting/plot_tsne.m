% t-SNE visualize_script 
% Don't Run from Main
%WARNING: Needs lots of RAM !

% uncomment depending on what you want to map:
% training data (no zero shot classes) or test data (includes zero shot classes)
%  load('mappedTrainData.mat');
%load('./gauss_cifar10_acl_cat_truck/mappedTestImages.mat');
if outputPath 
    fprintf('Output Path %s',outputPath); 
else
    outputPath = './gauss_cifar10_small_acl_cat_truck_0.0001';
end

load([outputPath '/mappedTestImages.mat']);
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

%% mapping everything in same space
    %t = tsne([mappedTestImages{2} wordTable]');
figure;
num_seen_cats = 8;

% [C,t,l]= pca([mappedTestImages{1} mappedTestImages{2} mappedTestImages{3} ...
%     mappedTestImages{4} mappedTestImages{5} mappedTestImages{6} ...
%     mappedTestImages{7} mappedTestImages{8} wordTable]','NumComponents',2);
t = tsne([mappedTestImages{1} mappedTestImages{2} mappedTestImages{3} mappedTestImages{4} mappedTestImages{5} mappedTestImages{6}  mappedTestImages{7} mappedTestImages{8} wordTable]');
    
mappedWordTable_t = t(numImages*num_seen_cats+1:end, :);
sym_array = '+o*.xsd^v><';
scatter(mappedWordTable_t(:,1), mappedWordTable_t(:,2), 200, 'd', 'k', 'filled');
hold on;
for i=1:num_seen_cats
    mappedX_t = t((i-1)*numImages +1 : (i-1)*numImages + numImages,:);
    cat_id = nonZeroCategories(i);

    idx = find(Y~=cat_id);
    p1 = scatter(mappedX_t(idx,1), mappedX_t(idx,2),3,'bo');
    idx = find(Y==cat_id);
    p2 = scatter(mappedX_t(idx,1), mappedX_t(idx,2),4, 'r+');
    %gscatter(mappedX_t(idx,1), mappedX_t(idx,2), label_names(Y_bin), [], '+', 8);
    hold on;
    %gscatter(mappedX_t(idx,1), mappedX_t(idx,2), label_names(Y_bin), [], 'o', 8);
    
%     Y_bin = (Y==cat_id);
%     Y_bin = (Y_bin* cat_id) + 1;
%     %unique(Y_bin)
%      syms = [sym_array(1), sym_array(max(Y_bin))]
%     %gscatter(mappedX_t(:,1), mappedX_t(:,2), label_names(Y_bin), [], sym_array, 8);
end
for i = 1:length(label_names)-1
    text(mappedWordTable_t(i,1),mappedWordTable_t(i,2),['  ' label_names{i+1}],'BackgroundColor',[.7 .9 .7]);        
end

legend([p1,p2],'Non-class Image','Class Image', 'Location','northeast');

axis off;
hold off;

%title('Confustion Matrix post Mapping training');
file_name = [outputPath '/SemanticWordSpace_seen.jpg'];
Image = getframe(gcf);
imwrite(Image.cdata, file_name);

%% 
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
