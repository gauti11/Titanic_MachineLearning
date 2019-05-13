function [J, grad] = LR_GDR(theta, X, y, lambda)
m = length(y); % number of training examples

% initializing the values
J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);

J_old = (1/m) * ( -y' * log(h) - (1-y)'*log(1 - h)) ;

theta_sum = sum(theta .* theta) - theta(1)*theta(1);

regression = ((lambda/(2*m))*(theta_sum)) ;

J = J_old + regression ;


grad_old =  (1/m) * X' *(h - y) ;

grad =  (1/m) * X' *(h - y) + ((lambda/m) * theta) ;

grad(1) = grad_old(1) ;


end
