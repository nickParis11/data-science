function [C,sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%%% create sigma
sigma_vec=zeros(1,8);
l=length(sigma_vec);
sigma_vec(1)=0.01;

for a=2:8;
if mod(a,2)==0;
mul=3;
else
mul=3+(1/3);
end;

sigma_vec(a)=sigma_vec(a-1)*mul;

end;

%sigma_vec

%%%create C

c_vec=zeros(1,8);
h=length(c_vec);
c_vec(1)=0.01;

for a=2:8;
if mod(a,2)==0;
mul=3;
else
mul=3+(1/3);
end;

c_vec(a)=c_vec(a-1)*mul;


end;

%c_vec

%%% create mixed matrix of 64 combinations of sigma and C

mix_vec=zeros((l*h),4);

indice=1;

for i=1:l
for j=1:h
mix_vec(indice,1)=sigma_vec(i);
mix_vec(indice,2)=c_vec(j);
mix_vec(indice,3)=indice;

indice=indice+1;

endfor;
endfor;

%%%%%%%%%%%%%% Train Model %%%%%%%%%%%%%%

for ind=1:size(mix_vec,1)
 
%%% to remove
%ind=8;
%mix_vec(:,4)=round(rand(64,1)*100);
%%% end to remove

sigma = mix_vec(ind,1);
C = mix_vec(ind,2);
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
pred=svmPredict(model, Xval);
pred;

acc=mean(double(pred~=yval));
ind
acc
mix_vec(ind,4)=acc;

endfor;

[i,ix]= min(mix_vec,[],1);
temp=ix(4); % index 4 because taking min of C or sigma's is useless

sigma=mix_vec(temp,1)
C=mix_vec(temp,2)
temp

mix_vec
save mix_vec.txt mix_vec

% =========================================================================

end
