function [theta] = updateParam(theta,grad, trainParams)
    lr = trainParams.lr;
    theta = theta - (lr*grad);
end