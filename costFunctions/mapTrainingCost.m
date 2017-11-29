function [cost, grad] = mapTrainingCost( theta, data, params )

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

% Find error signal terms
del3All = (ty .*w_min_h) - (1-ty) .*(lambda_penalty*(w_min_h)) ;

sparsityMult = params.beta * (-(params.sparsityParam ./ rhoHat) + (1 - params.sparsityParam) ./ (1 - rhoHat));
del2All = bsxfun(@plus, W{2}' * del3All, sparsityMult * 0.5) .* params.f_prime(a2All);

% Calculate gradients
W2grad = del3All * a2All' / numAllImages + lambda * W{2};
b2grad = sum(del3All, 2) / numAllImages + lambda * b{2};
W1grad = del2All * allImgs' / numAllImages + lambda * W{1};
b1grad = sum(del2All, 2) / numAllImages + lambda * b{1};

grad = [ W1grad(:); W2grad(:); b1grad(:); b2grad(:) ];

end
