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


temp1=X*theta;
%temp1

%temp2=sigmoid(temp1)

temp2=zeros(length(temp1),1);

for i=1:length(temp1);
temp2(i,1)=(1/(1+e^-temp1(i)));
end;


%temp2

temp4=length(theta);


%temp3=[theta(2,1):theta(temp4,1)]
%temp5=length(temp3);
%temp5
%(theta'*theta);


tosquare=theta(2:length(theta));
tosquare

J=((-y'*log(temp2)-(1-y)'*log(1-temp2))/m)+(((lambda/(2*m))*(tosquare'*tosquare)));


grad(1)=(temp2-y)'*X(:,1)/m;
%grad(1)
for i=2:length(grad)
grad(i)=((temp2-y)'*X(:,i)/m)+((lambda/m)*theta(i));

end;
%grad


J

%save output.txt grad J



% =============================================================

end
