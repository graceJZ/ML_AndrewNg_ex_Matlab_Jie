function [J,grad] = nnCostFunction(nn_params, ...
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
K=max(y);
Y=zeros(m,K);
for kk=1:m
    Y(kk,y(kk))=1;
end

a1 = [ones(m,1) X];%5000*401;
z2=a1*Theta1'; %5000*25;
a2=[ones(m,1) sigmoid(z2)];% 5000*26
z3=a2*Theta2';% 5000*10;
a3=sigmoid(z3);% 5000*10;

J=trace(-Y*log(a3')-(1-Y)*log(1-a3'));
J=J/m;

% Part 2: Implement the backpropagation algorithm to compute the gradients

sigma3=a3-Y; % 5000*10;
sigma2=(sigma3*Theta2(:,2:end)).*sigmoidGradient(z2);% 5000*25

delta2=zeros(size(Theta2));% 10*26
delta2=delta2+sigma3'*a2;


delta1=zeros(size(Theta1)); %25*401
delta1=delta1+sigma2'*a1;


Theta1_grad=delta1/m;
Theta2_grad=delta2/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
theta1=Theta1(:,2:end); % Theta1 is 25*401;
theta2=Theta2(:,2:end); % Theta2 is 10*26;
J=J+lambda*trace(theta1*theta1')/2/m;
J=J+lambda*trace(theta2*theta2')/2/m;

Theta1_grad=Theta1_grad+lambda*[zeros(size(Theta1,1),1) Theta1(:,2:end)]/m;
Theta2_grad=Theta2_grad+lambda*[zeros(size(Theta2,1),1) Theta2(:,2:end)]/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
% end
