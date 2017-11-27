% Total training examples are 50000, makes data 1/5 of original data for experimentation
function splitData(trainX,trainY) 
T =[];
Y = [];
for i = 1:3
    j = find(trainY == i-1);
    t  = trainX(:,j);
    y = trainY(j,:);
    r = randi(5000,[1,1000]);
    T = [T, t(:,r)];
    Y = [Y; y(r)];
end

save('data_new.mat','T','Y');
