function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)


%%%%%%%%%%%%%%%

%problémes enregistrés au 15/11/13 computecost theta(0;0) et gradientdescent outputent un J différent pour la première itération ( debug : mmettre un trace de computecost sur itération 1 ) 



%%%%%%%%%%%%%%%%


%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 4);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

for i=1:size(X)(2)


D=X*theta;

E=D-y;

F=E'*X(:,i);

G=F/m;

H=G*alpha;

%H

%I=E'*X(:,2);

%J=I/m;

%K=J*alpha;
%K

thetaTemp(i)=theta(i)-H;
%thetaTemp1=theta(2)-K;

end

for j=1:size(X)(2)

theta(j)=thetaTemp(j);



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter,1) = computeCost(X, y, theta);
    J_history(iter,2) = theta(1)	;	
    J_history(iter,3) = theta(2)	;
    J_history(iter,4) = theta(3)	;

J_history;

iter=iter+1;


end

save output.txt J_history 

end
