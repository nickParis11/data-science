function [error_val2, error_train2] = testLRCF (X,y,Xval,yval) ;

lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';


error_val2=zeros((size(lambda_vec)(1)),1);

error_train2=zeros(size(error_val2));

X=[ones(size(X,1),1) X];


Xval=[ones(size(Xval,1),1) Xval];


%theta2=[1;1];

%theta2

for z=1:length(lambda_vec)

theta2=trainLinearReg(X,y,lambda_vec(z));

error_val2(z)=linearRegCostFunction(Xval,yval,theta2,lambda_vec(z));

error_train2(z)=linearRegCostFunction(X,y,theta2,lambda_vec(z));


endfor;

i=lambda_vec(1);
j=lambda_vec(2);
J2=linearRegCostFunction(X, y, theta2, i)
J3=linearRegCostFunction(X, y, theta2, j)
error_train2
error_val2

end;