function [J, grad] = lrCostFunction(theta, X, y, lambda)

	m = length(y); % number of training examples
	J = 0;
	grad = zeros(size(theta));
	temp = theta;
	temp(1) = 0;
	h = sigmoid(X*theta);
	J = (-y'*log(h)-(1-y)'*log(1-h))/m+lambda/2/m*temp'*temp;
	grad = X'*(h-y)/m + lambda*temp/m;
	grad = grad(:);

end
