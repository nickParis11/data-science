function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).



z=1./(1+e.^-z);
z1=zeros(size(z));
z1=1-z;

%size(z)
%size(z1)
l=size(z)(2)*size(z)(1);
%l

for u=1:l;
%u
g(u)=z(u).*z1(u);
end;

%g=[1 g];
%size(g)
%size(g)
%g=z'*z1;











% =============================================================




end
