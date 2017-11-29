function [ guessedCategoriesDebug, results ] = mapDoEvaluate( images, categories, cat_id, nonZeroCategories, originalCategoryNames, testCategoryNames, testWordTable, theta, trainParams, doPrint )

numImages = size(images, 2);
numCategories = size(testWordTable, 2);
numSeenCat = length(nonZeroCategories); 
dist = zeros(numCategories,numImages);
% to make non zero categories as infinity, so that those are not guesseed
% whiel plotting confusiton matrix
dist(not(ismember(1:10,nonZeroCategories)),:) = inf;
% Feedforward
for i =1:numSeenCat
    cat_id = nonZeroCategories(i);
    %find projection of every image using respective theta
    mappedImages = mapDoMap(images, theta{i}, trainParams);
    w = testWordTable(:,i);%Dx1
    %w = repmat(w,1,numImages);
    dist(cat_id,:) = slmetric_pw(w,mappedImages,'eucdist');  %8xN
end

%dist = slmetric_pw(testWordTable, mappedImages, 'eucdist');
[ ~, guessedCategories ] = min(dist);

% map categories from originalCategoryNames to testCategoryNames
%mappedCategorySet = zeros(1, length(originalCategoryNames));
%  for i = 1:length(originalCategoryNames)
%     mappedCategorySet(i) = find(strcmp(originalCategoryNames{i}, testCategoryNames));
%  end
% % mappedCategories = arrayfun(@(x) mappedCategorySet(x), categories);

%guessedCategoriesDebug = [ dist; mappedCategories'; guessedCategories ];

% Calculate scores
confusion = zeros(numSeenCat, numSeenCat);
for i = 1:numSeenCat
    actual = nonZeroCategories(i);
	%how many labels are predicted as 1..etc
    guessesForCategory = guessedCategories(categories == actual);
    for j = 1:numSeenCat
        guessed = nonZeroCategories(j);
        confusion(i, j) = sum(guessesForCategory == guessed);
    end
end

truePos = diag(confusion); % true positives, column vector
results.accuracy = sum(truePos) / numImages;
t = truePos ./ sum(confusion, 2);
results.avgPrecision = mean(t(isfinite(t), :));
t = truePos' ./ sum(confusion, 1);
results.avgRecall = mean(t(:, isfinite(t))); %Filter out NaNs if present

if doPrint == true
    disp(['Accuracy: ' num2str(results.accuracy)]);
    disp(['Averaged precision: ' num2str(results.avgPrecision)]);
    disp(['Averaged recall: ' num2str(results.avgRecall)]);
    displayConfusionMatrix(confusion, testCategoryNames);
end

end

