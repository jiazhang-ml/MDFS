% This is an example file on how the MDFS [1] program could be used.

% [1] J. Zhang, Z. Luo, C. Li, C. Zhou, S. Li: 
% Manifold regularized discriminative feature selection for multi-label learning, Pattern Recognition, 2019, 95: 136-150.

% Please feel free to contact me (zhangjia_gl@163.com), if you have any problem about this program.

clc; clear; 
addpath(genpath('.\'))
load('Scene_data.mat')

% Calucate some statitics about the data
[num_train, num_label] = size(Y_train); [num_test, num_feature] = size(X_test);

pca_remained = round(num_feature*0.95);

% Performing PCA
all = [X_train; X_test]; 
ave = mean(all);
all = (all'-concur(ave', num_train + num_test))';

covar = cov(all); covar = full(covar);

[u,s,v] = svd(covar);

t_matrix = u(:, 1:pca_remained)';
all = (t_matrix * all')';

X_train = all(1:num_train,:); X_test = all((num_train + 1):(num_train + num_test),:);
    
para.alpha = 1; para.beta = 1; para.gamma = 100; para.k = num_label - 1;

% Running the MDFS procedure for feature selection
t0 = clock;
[ W, obj ] = MDFS( X_train, Y_train, para );
time = etime(clock, t0);

[dumb idx] = sort(sum(W.*W,2),'descend'); 
feature_idx = idx(1:pca_remained);

% The default setting of MLKNN
Num = 10;Smooth = 1;  

% Train and test
% If you use MLKNN as the classifier, please cite the literature [2]
% [2] M.-L. Zhang, Z.-H. Zhou:
% ML-KNN: A lazy learning approach to multi-label learning. Pattern Recognition 2007, 40(7): 2038-2048.
for i = 1:pca_remained
    fprintf('Running the program with the selected features - %d/%d \n',i,pca_remained);
    
    f=feature_idx(1:i);
    [Prior,PriorN,Cond,CondN]=MLKNN_train(X_train(:,f),Y_train',Num,Smooth);
    [HammingLoss,RankingLoss,Coverage,Average_Precision,macrof1,microf1,Outputs,Pre_Labels]=...
        MLKNN_test(X_train(:,f),Y_train',X_test(:,f),Y_test',Num,Prior,PriorN,Cond,CondN);
    
    HL_MDFS(i)=HammingLoss;
    RL_MDFS(i)=RankingLoss;
    CV_MDFS(i)=Coverage;
    AP_MDFS(i)=Average_Precision;
    MA_MDFS(i)=macrof1;
    MI_MDFS(i)=microf1;
end
