function [ W, obj ] = MDFS( X_train, Y_train, para )
%% function discription
% ----------------------------------------------------------------------
% Function for multi-label feature selection
% Jia Zhang
% Inputs:   
%          X_train--A n x m array, the i-th instance is stored in X_train(i,:)
%          Y_train--A n x q array, q is the number of possible labels, Y_train(i,j) is 1 if the i-th instance has the j-th label, and -1 otherwise
%
%          para--the set of parameters
%
% Outputs:  
%          W--A d x q array, d is the number of features, coefficient matrix for feature selection
%
%          obj--The objective values
%
% ----------------------------------------------------------------------

% Calucate some statitics about the data
[num_train, num_feature] = size(X_train); num_label = size(Y_train, 2);

%L, L0, and H
L = Laplacian_GK(X_train', para); L0 = Laplacian_GK(Y_train, para);

H = eye(num_train) - 1 / num_train * ones(num_train, 1) * ones(num_train, 1)';

%Initialize W
W = rand(num_feature, num_label); 


iter = 1; obji = 1;
while 1
    
    %Update F--------------------------------------------------------------
    A = L + para.alpha * H + para.alpha * eye(num_train);
    B = para.beta * L0;
    C = - para.alpha * H * X_train * W - para.alpha * Y_train;
    F = lyap(A, B, C);  
    
    %Update W--------------------------------------------------------------
    d = 0.5./sqrt(sum(W.*W, 2) + eps);
    D = diag(d);
    W = (para.alpha * X_train' * H * X_train + para.gamma * D + eps*eye(num_feature)) \ (para.alpha * X_train' * H * F); 
    
    obj(iter) =  trace(F'*L*F) + para.alpha*(norm((H*X_train*W - H*F), 'fro'))^2 + para.alpha*(norm((F - Y_train), 'fro'))^2 ...
        + para.beta*trace(F*L0*F') + para.gamma * sum(sqrt(sum(W.*W,2)+eps));

    cver = abs((obj(iter) - obji)/obji);
    obji = obj(iter);
    iter = iter + 1;
    if (cver < 10^-3 && iter > 2) , break, end
    
end

end

