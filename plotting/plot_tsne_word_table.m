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

load('./word_data/acl/cifar10/wordTable.mat')
% load label names
load('image_data/images/cifar10/meta.mat');

% load word table
load('word_data/acl/cifar10/wordTable.mat');
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
t = tsne([wordTable]');
    
mappedWordTable_t = t;
sym_array = '+o*.xsd^v><';
scatter(mappedWordTable_t(:,1), mappedWordTable_t(:,2), 200, 'd', 'k', 'filled');
hold on;

for i = 1:length(label_names)-1
    text(mappedWordTable_t(i,1),mappedWordTable_t(i,2),[' ' label_names{i+1}],'BackgroundColor',[.7 .9 .7]);        
end


axis off;
hold off;

%title('Confustion Matrix post Mapping training');
file_name = [outputPath '/SemanticWords.jpg'];
Image = getframe(gcf);
imwrite(Image.cdata, file_name);

