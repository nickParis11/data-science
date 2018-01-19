function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 3);
computeCost(X,y,theta)

for iter=1:num_iters;
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

iter
theta
%theta(1)
%theta(2)
D=X*theta;

E=D-y;

F=E'*X(:,1);

G=F/m;

H=G*alpha;

H

I=E'*X(:,2);

J=I/m;

K=J*alpha;
K

thetaTemp0=theta(1)-H;
thetaTemp1=theta(2)-K;


%thetaTemp1=theta(1)-alpha*(E'*X(:,1)/m)
%thetaTemp2=theta(2)-alpha*(E'*X(:,2)/m)
theta(1)=thetaTemp0;
theta(2)=thetaTemp1;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter,1) = computeCost(X, y, theta);
    J_history(iter,2) = theta(1)	;	
    J_history(iter,3) = theta(2)	;

J_history;

iter=iter+1;
end

save output.txt J_history 

end
