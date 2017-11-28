% Total training examples are 50000, makes data 1/5 of original data for experimentation
%** update paths based on location of originally extracted features
load('/Volumes/W/train.mat')
scale_down_factor = 0.02;%10 percent
disp('images per class');
no_of_imgs = 5000*scale_down_factor;
disp(no_of_imgs);
disp('Loading Data');
tempX =[];
tempY = [];

for i = 1:10
    j = find(trainY == i-1);
    
    indices = randi(5000,[1,no_of_imgs]);
    j = j(indices); %only selct few out of 5000
    tempY = [tempY ; trainY(j)];
    tempX = [tempX, trainX(:,j)];
%     T = [T, t(:,r)];
%     Y = [Y; y(r)];
end
trainX = tempX;
trainY = tempY;
save('train.mat','trainX','trainY');

load('/Volumes/W/test.mat')
tempX =[];
tempY = [];
no_of_imgs = 1000*scale_down_factor;

for i = 1:10
    j = find(testY == i-1);
    
    indices = randi(1000,[1,no_of_imgs]);
    j = j(indices); %only selct few out of 5000
    tempY = [tempY ; testY(j)];
    tempX = [tempX, testX(:,j)];
%     T = [T, t(:,r)];
%     Y = [Y; y(r)];
end
testX = tempX;
testY = tempY;
save('test.mat','testX','testY');