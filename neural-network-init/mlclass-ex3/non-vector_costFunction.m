function [J, grad] = NV_costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

temp1=X*theta;
%temp1

%temp2=sigmoid(temp1)

temp2=zeros(length(temp1),1);

for i=1:length(temp1);
temp2(i,1)=(1/(1+e^-temp1(i)));
end;


%temp2

%temp2=e^-(temp1);
%temp2=1/(1+e^-(temp1));


%temp3=-y'*log(temp1);


%y2=1-y; %!!!!!
%y2

J=(-y'*log(temp2)-(1-y)'*log(1-temp2))/m;
J

%temp3=(temp2-y)'*X(:,2)/m;
%temp3

for i=1:length(grad)
grad(i)=(temp2-y)'*X(:,i)/m;
end;
grad


save output4.txt grad J



% =============================================================

end
