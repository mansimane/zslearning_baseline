function [cost] = calcCost( theta, data, params )

[ W, b ] = stack2param(theta, params.decodeInfo);

cat_id = data.cat_id;
ty = (cat_id == data.categories)';
lambda_penalty = params.lambda_penalty;

numImages = size(data.imgs, 2);
allImgs = [ data.imgs ];
numAllImages = size(allImgs, 2);

a2All = params.f(bsxfun(@plus, W{1} * allImgs, b{1}));
hAll = bsxfun(@plus, W{2} * a2All, b{2});
w = data.wordTable(:, cat_id);  
w = repmat(w, 1,numImages ); %%50x760 just for computatioinal convinience, can be made one d
rhoHat = (sum(a2All, 2) / numAllImages + 1) / 2; % scale to [0,1] since we're using tanh
lambda = params.lambda;
reg = 0.5 * lambda * (sum(sum(W{1} .^ 2)) + sum(sum(W{2} .^ 2)) + sum(b{1} .^ 2) + sum(b{2} .^ 2));
sparsity = params.beta * sum(params.sparsityParam * log(params.sparsityParam ./ rhoHat) ...
    + (1 - params.sparsityParam) * log((1 - params.sparsityParam) ./ (1 - rhoHat)));

w_min_h = hAll - w; %w_min_h =DxN, ty = 1xN
J = sum( ty.*  sum((w_min_h).^2)) - sum(lambda_penalty *((1-ty) .* sum((w_min_h).^2))) ;

cost = 0.5 * (1 / (numImages) * J ) + sparsity + reg;


end
