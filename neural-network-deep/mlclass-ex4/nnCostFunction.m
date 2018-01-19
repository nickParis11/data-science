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

%%%%%% A4 transform y to ymat de num_labels=10 labels ( or output unit and m training examples

ymat=zeros(m,num_labels);
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
%b

%sum(sum(sqT2))

J=J+b;

%%%%%%  A7 : Backpropagation

%%% A7_0 : Set DELTA DEF 

%size(X)
%la1

%DELTAD0=zeros(1,hidden_layer_size);
DELTAD0=zeros(hidden_layer_size,input_layer_size+1);
%DELTAD1=zeros(1,num_labels);
DELTAD1=zeros(num_labels,hidden_layer_size+1);


%%% A7_1 : feedforward

for t=1:m;
%t % tracer boocle for
% setting layer sizes

la1=zeros(1,size(X)(2));

la2=zeros(1,hidden_layer_size);
la3=zeros(1,num_labels);

%disp("size la1");
%size(la1)

la1=X(t,:); % choosing X(t) to be layer 1 (1x401)

la2=sigmoid(la1*Theta1');
%size(la2)
la2=[1 la2];

la3=sigmoid(la2*Theta2');


%%% A7_2 Delta computation and resize layers +  delta's


delta2=zeros(size(la3)); % on pourrait mettre size(num_labels)
delta2=la3-ymat(t,:);


%delta2

delta2=delta2';%%%%%%



delta1=zeros(size(la2))';%%%%% added the transpose

la1=la1'; %%%%%
la2=la2'; %%%%%
la3=la3'; %%%%%


%%%%% delta1=zeros(size(la2));

delta1=Theta2'*delta2;

%disp("size(delta1)")
%size(delta1)

delta1=delta1(2:end); % choice not to take bias unit delta0 ( delta(1) in octave since vectors indexing starts @ 1) for backprop of delta's


delta1=delta1'.*sigmoidGradient(la1'*Theta1');%%%%% 
delta1=delta1';

%disp("size(delta1) après sigmoid")
%size(delta1)


%%%%%delta1=delta2*Theta2.*sigmoidGradient(la1*Theta1');% why is it noted in the exercise that we should transpose Theta2 ???




%%% A7_3 Accumulate delta's


%DELTAD1=DELTAD1+(delta2*la3'); 
%%%%%DELTAD1=DELTAD1+(delta2'*la2);
DELTAD1=DELTAD1+(delta2*la2'); %%%%%

%DELTAD0=DELTAD0+(delta1*la2(2:end)');
%%%%%DELTAD0=DELTAD0+(delta1'*la1);
DELTAD0=DELTAD0+(delta1*la1'); %%%%%

%%% Display some t inside the loop for debugging purposes 


%if mod (t,4088) == 0 ;
%t
%delta1
%disp("DELTA DEF 1 = ");
%DELTAD1
%disp("DELTA DEF 2 = ");
%DELTAD2

%save output_2.txt DELTAD1 DELTAD0

%end;

%la1
%size(la1)
%size(la2)
%size(la3)
%size(l3)
%la3


%size(delta2)
%disp("ymat = ");
%la3
%ymat(t,:)
%delta2



end; % end for

%disp("DELTA DEF 1 = ");
%DELTAD1
%disp("DELTA DEF 2 = ");
%DELTAD2

%save output_3.txt DELTAD1 DELTAD0

D1=DELTAD0./m;
D2=DELTAD1./m;

save output_3.txt D1 D2 % UnREg Gradients

%%% A8 adding Regularizations to gradients


%D1(i,j)=D1(i,j)+(lambda/m).*Theta1(i,j)

D1Temp=zeros(size(D1)(1),size(D1)(2)-1);
D1Temp=D1(:,2:end);
D1Temp=D1Temp+(lambda/m).*Theta1(:,2:end);
D1=[D1(:,1) D1Temp];

D2Temp=zeros(size(D2)(1),size(D2)(2)-1);
D2Temp=D2(:,2:end);
D2Temp=D2Temp+(lambda/m).*Theta2(:,2:end);
D2=[D2(:,1) D2Temp];


save debug.txt D1Temp D1 D2Temp;


Theta1_grad=D1;
Theta2_grad=D2;




save output_4.txt Theta1_grad Theta2_grad DELTAD0 DELTAD1
%save output.csv Theta1_grad Theta2_grad


%-------------------------------------------------------------
% nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

% nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda) 

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%grad
save output_5.txt grad
end














%%%%%%%%%% OLD A7 code

%la2=zeros(1,hidden_layer_size); % attention la2 may be of 1xhidden_layer_size
%la3=zeros(m,num_labels);

%size(la2)
%size(la3)

%la1=X(1,:);
%la1

%for t=1:m;
%la1=X(t,:);

%la2=sigmoid(X*Theta1');
%la2=[1 la2];
%la3=sigmoid(la2*Theta2');
%if mod(t,2044) == 0;
%t
%la1(t,:)
%la3(t,:)

%end;

%end;

