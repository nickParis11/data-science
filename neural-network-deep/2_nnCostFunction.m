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



%%%% A1 : Roll out Theta's

lenT1=hidden_layer_size*(input_layer_size+1);
%lenT1

Theta1=reshape(nn_params(1:lenT1),hidden_layer_size,input_layer_size+1);
%size(Theta1)
%Theta1

Theta2=reshape(nn_params((lenT1+1):end),num_labels,(hidden_layer_size+1));
%size(Theta2)

% input_layer_size
% hidden_layer_size
% num_labels


% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


save output.txt Theta1 Theta2 

%%%%%% A2 : Set Layers
% layers 1 = X , set layer 2 and 3 ( hidden and output )

l2=zeros(m,hidden_layer_size);
%size(l2)

l3=zeros(m,num_labels);
%size(l3)

%%%%%% A3 : feedforward

X=[ones(m,1) X];
%size(X)

l2=sigmoid(X*Theta1');
%l2(1,:)

l2=[ones(m,1) l2 ];
%fprintf('size de l2 /n');
%size(l2)

l3=sigmoid(l2*Theta2');
%fprintf('size de l3 /n');
%size(l3)
%fprintf('\n exemple d'output.../n');
%fprintf('une ligne de l3 /n');
%l3(4001,:)
%log(l3(4001,:))

%%%%%% A4 transform y to ymat de num_labels=10 labels

ymat=zeros(5000,10);
for i=1:m;
%if y(i)==10;
%i2=0;
%else
i2=y(i);
ymat(i,i2)=1;
%end;
end;

%ymat(503,:)     
%ymat(4001,:) 

%%%%%% A5 Compute cost function loop

% init error et temp error

err=0;
terr=0;

for j=1:m
	terr=0;
	for i=1:num_labels
	
	%log(l3(j,i))	
	terr=-ymat(j,i)*log(l3(j,i))-(1-ymat(j,i))*log(1-(l3(j,i)));
	err=err+terr;

	%terr
	%i

end;

end;


J=err/m;
%J

%%%%%% A6 : Add regularization
%%%%%%%%%%%%%% A6_1 : Square Theta's

% input_layer_size
% hidden_layer_size
% num_labels


sqT1=Theta1.^2;
sqT1=sqT1(:,2:input_layer_size+1);

sqT2=Theta2.^2;
sqT2=sqT2(:,2:hidden_layer_size+1);
%size(sqT2)

b=(sum(sum(sqT2))+ sum(sum(sqT1)))* (lambda/(2*m));
b

%sum(sum(sqT2))

J=J+b

%%%%%%  A7 : Backpropagation
%
%%



% 
-------------------------------------------------------------
% nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
