function [sim] = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%
w=x1-x2;
simT1=(w.^2);
simT2=sqrt(simT1);
simT3=simT2.^2;
simT4=sum(simT3/(2*(sigma.^2)));
sim=exp(-simT4);

disp("gk");

% one line implementation
%simT=exp(-1*sum(((sqrt((x1-x2).^2)).^2)/(2*(sigma.^2))))

%simT


% =============================================================
    
end
