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
 %t = tsne([mappedTestImages{1} mappedTestImages{2} mappedTestImages{3} mappedTestImages{4} mappedTestImages{5} mappedTestImages{6}  mappedTestImages{7} mappedTestImages{8} wordTable]');
    
mappedWordTable_t = t(numImages*num_seen_cats+1:end, :);
sym_array = '+o*.xsd^v><';
 idx_z1 = find(Y== zeroCategories(1));
 idx_z2 = find(Y== zeroCategories(2));

for i=1:num_seen_cats
    mappedX_t = t((i-1)*numImages +1 : (i-1)*numImages + numImages,:);
    cat_id = nonZeroCategories(i);    
    %Index finding Can be moved outside loop
    p3 = scatter(mappedX_t(idx_z1,1), mappedX_t(idx_z1,2),3,'b*');
    
    idx_z2 = find(Y== zeroCategories(2));
    p4 = scatter(mappedX_t(idx_z2,1), mappedX_t(idx_z1,2),3,'r*');
    %gscatter(mappedX_t(idx,1), mappedX_t(idx,2), label_names(Y_bin), [], '+', 8);
    hold on;
    %gscatter(mappedX_t(idx,1), mappedX_t(idx,2), label_names(Y_bin), [], 'o', 8);
    
end
scatter(mappedWordTable_t(:,1), mappedWordTable_t(:,2), 150, 'd', 'k', 'filled');
hold on;
for i = 1:length(label_names)-1
    text(mappedWordTable_t(i,1),mappedWordTable_t(i,2),['' label_names{i+1}],'BackgroundColor',[.7 .9 .7]);        
end

axis off;
hold off;

legend([p3,p4],'Zero Shot category Cat','Zero Shot category Truck', 'Location','northeast');
axis off;
hold off;
set(gcf,'paperunits','centimeters')
set(gcf,'papersize',[30,25]) % Desired outer dimensionsof figure
set(gcf,'paperposition',[0,0,30,25]) % Place plot on figure

%title('Confustion Matrix post Mapping training');
file_name = [outputPath '/SemanticWordSpace_zero_shot.jpg'];
Image = getframe(gcf);
imwrite(Image.cdata, file_name);

%% Plot both
figure;
sym_array = '+o*.xsd^v><';
 idx_z1 = find(Y== zeroCategories(1));
 idx_z2 = find(Y== zeroCategories(2));

for i=1:num_seen_cats
    mappedX_t = t((i-1)*numImages +1 : (i-1)*numImages + numImages,:);
    cat_id = nonZeroCategories(i);

     idx = find(Y~=cat_id);
    p1 = scatter(mappedX_t(idx,1), mappedX_t(idx,2),3,'co');
     idx = find(Y==cat_id);
    p2 = scatter(mappedX_t(idx,1), mappedX_t(idx,2),4, 'r+');
        %Index finding Can be moved outside loop
    p3 = scatter(mappedX_t(idx_z1,1), mappedX_t(idx_z1,2),3,'b*');
    
    idx_z2 = find(Y== zeroCategories(2));
    p4 =scatter(mappedX_t(idx_z2,1), mappedX_t(idx_z1,2),3,'m*');
    %gscatter(mappedX_t(idx,1), mappedX_t(idx,2), label_names(Y_bin), [], '+', 8);
    hold on;
    %gscatter(mappedX_t(idx,1), mappedX_t(idx,2), label_names(Y_bin), [], 'o', 8);
    end
scatter(mappedWordTable_t(:,1), mappedWordTable_t(:,2), 150, 'd', 'k', 'filled');
hold on;
for i = 1:length(label_names)-1
    text(mappedWordTable_t(i,1),mappedWordTable_t(i,2),[ label_names{i+1}],'BackgroundColor',[.7 .9 .7]);        
%    text(mappedWordTable_t(i,1),mappedWordTable_t(i,2),['  ' label_names{i+1}]);        
end

legend([p1,p2,p3,p4],'Non-class Image','Class Image','Zero Shot cat. 1','Zero Shot cat. 2', 'Location','northeast');
axis off;
hold off;
set(gcf,'paperunits','centimeters')
set(gcf,'papersize',[30,55]) % Desired outer dimensionsof figure
set(gcf,'paperposition',[0,0,30,55]) % Place plot on figure

%title('Confustion Matrix post Mapping training');
file_name = [outputPath '/SemanticWordSpace_both_shot.jpg'];
Image = getframe(gcf);
imwrite(Image.cdata, file_name);

