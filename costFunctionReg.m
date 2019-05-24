function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

predicted_y = sigmoid(X*theta);
%lambda
%for iter = 1:size(X,2)
  
 %grad(i) =grad(i)+((1/m)*(sum((predicted_y-y).*X(:,i)))+lambda*grad(i));
 %i=i+1;
gradient_parameter = [0;theta(2:end)];
%grad =((predicted_y - y)' * X / m)' + lambda .* theta .* [0; ones(length(theta)-1, 1)] ./ m ;
grad =((predicted_y - y)' * X / m)' + lambda*(gradient_parameter.^2)./ m ;
J = (sum(-y.* log(predicted_y) - (1-y).*log(1-predicted_y))/m)+ (lambda*sum(theta(2:end).^2))/(2*m);
%J = (sum(-y .* log(predicted_y) - (1 - y).*log(1 - predicted_y)) / m) + lambda * sum(theta(2:end).^2) / (2*m);





% =============================================================

end
