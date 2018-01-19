function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

X1=rand(299,2)*100;
%X=X1;

%%% for each centroid
for j=1:K

%%% create vector of indices refering to the dataset examples which are assigned to that cluster ( Hereafter PopK )
B=find(idx==j);
%size(B)
%B(1:5)

%%% create empty matrix in order to store the values of every examples of PopK
A2=zeros(length(B),n);
%size(A2)

%temp=length(find(idx==j));
%A=[(find(idx==j)) ones(temp,1)];
%%size(A)


l=size(A2,1);

%%% for every element of popK assign his n values from the dataset into the empty matrix

for i=1:l;


A2(i,:)=X(B(i),:);


endfor;

%%%for every dim/column of the populated matrix, get the mean of that column in order to be a new coordonate ofthis centroid

for h=1:n

%centroids(j,h)=sum(A2(:,h))/l; 
centroids(j,h)=mean(A2(:,h));

endfor;




endfor;

centroids





% =============================================================


end

