function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%c=X(1:10,:); test visualisation de X
%c

temp2=1./(1+e.^-(X*theta));
%temp2

%temp4=temp2(1:10,:); %test sur 400 unités avant de vectoriser une multiplication de matrice avec transposition de l'une d'entre elle.
%temp4
%temp5=y(1:10);
%temp5
%temp6=temp4'*temp5 % pre test de temp3
%temp6

%temp3= temp2'*y; % pre test que J va bien varier si on fait varier les valeurs de y
%temp3

% ancienne version avec multiplication non vectorisée
%temp2=zeros(length(temp1),1); 
%for i=1:length(temp1);
%temp2(i,1)=(1/(1+e^-temp1(i)));
%end;

J=((-y'*log(temp2)-(1-y)'*log(1-temp2))/m)+(((lambda/(2*m))*(theta(2:end)'*theta(2:end))));
J

%for i=1:length(grad)% version non vectorisée de calcul du gradient
%grad=(temp2-y)'*X(:,i)/m;
%end;

temp_grad=theta.*lambda./m; % test temp grad
%temp_grad
grad(1)=((X(:,1)'*(temp2-y)))./m; % faire cas grad 0
fprintf('Grad 1 = \n')
grad(1)

grad(2:end)=(X(2:end)'*(temp2(2:end)-y(2:end)))./m; % cas grad > 0

grad 

save output3_1.txt grad J

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% =============================================================

grad = grad(:);

end
