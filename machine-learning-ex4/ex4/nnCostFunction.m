function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
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

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
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


a1 = X;
a1 = [ones(size(a1,1),1) a1];
a2 = sigmoid(a1*Theta1');
a2 = [ones(size(a1,1),1) a2];
hypo = sigmoid(a2*Theta2');

Y = zeros(size(y,1),num_labels);
for i=1:size(y,1),
    row = y(i);
    Y(i,row) = 1;
end

sum=0;
for i=1:size(Y,1),
    val = 0; 
    for j=1:size(Y,2),
        val = val - 1/m*((Y(i,j)*log(hypo(i,j))) + (1-Y(i,j))*log(1-hypo(i,j)));
    end
    sum = sum + val;
end

% val = Y.*hypo

% J = -1/m*(sum(Y,2)'*log(sum(hypo,2)) + sum((1-Y),2)'*log(sum(1-hypo,2)))
J = sum;

% regularization part
Theta1_new = Theta1;
Theta1_new(:,1) = 0;
Theta2_new = Theta2;
Theta2_new(:,1) = 0;

sum=0;
for i=1:size(Theta1_new,1),
    val=0;
    for j=1:size(Theta1_new,2),
        val = val + Theta1_new(i,j)^2;
    end
    sum = sum + val;
end

reg_part1 = sum;

sum=0;
for i=1:size(Theta2_new,1),
    val=0;
    for j=1:size(Theta2_new,2),
        val = val + Theta2_new(i,j)^2;
    end
    sum = sum + val;
end

reg_part2 = sum;
J = J + lambda/(2*m)*(reg_part1 + reg_part2);

% -------------------------------------------------------------
delta_2 = zeros(size(Theta2));
delta_1 = zeros(size(Theta1));
for i=1:size(Y,1),
    % feed forword
    a1 = [1 X(i,:)];
    z2 = Theta1*a1';
    a2 = [1; sigmoid(z2)(:) ];
    z3 = Theta2*a2;
    a3 = sigmoid(z3);
    size(a3);

    % error propogation
    size(Y(i,:));
    err3 = a3-Y(i,:)';
    err2 = Theta2'*err3.*sigmoidGradient([1; z2]);
    err2(1) = 0;
    size(err2);
    size(a1);
    %err1 = theta1'*err2.*sigmoidGradient(z1);
    %err1(1) = 0;
    err_wof = err2(2:size(err2,1),1);

    delta_2 = delta_2 + err3*a2';
    delta_1 = delta_1 + err_wof*a1;
end

Theta2_grad = 1/m*delta_2 + lambda/m*(Theta2_new);
Theta1_grad = 1/m*delta_1 + lambda/m*(Theta1_new);

size(Theta2_grad);
size(Theta1_grad);


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
