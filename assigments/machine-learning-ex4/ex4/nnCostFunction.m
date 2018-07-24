function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Y = zeros(m, num_labels);
for i=1:m
  Y(i,y(i)) = 1;
end


X = [ones(size(X,1),1) X];
a2 = sigmoid(X * Theta1');
a2 =[ones(size(a2,1),1) a2];
a3 = sigmoid(a2 * Theta2');
J = (1/m)*sum(sum((-Y).*log(a3) - (1-Y).*log(1-a3), 2)) ;
T_1 = Theta1(:,2:end);
T_2 = Theta2(:,2:end);
T_1 = T_1 .^2;
T_2 = T_2 .^2;
J = J + lambda/2/m*(sum(sum(T_1)) + sum(sum(T_2)));


largeDelta1 = zeros(size(Theta1));
largeDelta2 = zeros(size(Theta2));
for i = 1:m,	
	delta3 = (a3(i,:) - Y(i,:))';
	delta2 = (Theta2'*delta3).*(a2(i,:)'.*(1-a2(i,:)'));

	largeDelta1 = largeDelta1 + (delta2*X(i,:))(2:end,:);
	largeDelta2 = largeDelta2 + (delta3*a2(i,:));
end;


Theta1_grad = largeDelta1*1/m;

Theta2_grad = largeDelta2*1/m;


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
temp =Theta1;
temp (:,1)=0;
tmp = Theta2;
tmp(:,1)=0;
temp = lambda/m*[temp(:) ;tmp(:)];
grad = grad + temp;

end
