function [HammingLoss,RankingLoss,Coverage,Average_Precision,macrof1,microf1,Outputs,Pre_Labels]=MLKNN_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN)

    [num_class,num_training]=size(train_target);
    [num_class,num_testing]=size(test_target);
    
%Computing distances between training instances and testing instances
    dist_matrix=zeros(num_testing,num_training);
    for i=1:num_testing
        if(mod(i,100)==0)
            disp(strcat('computing distance for instance:',num2str(i)));
        end
        vector1=test_data(i,:);
        for j=1:num_training
            vector2=train_data(j,:);
            dist_matrix(i,j)=sqrt(sum((vector1-vector2).^2));
        end
    end

%Find neighbors of each testing instance
    Neighbors=cell(num_testing,1); %Neighbors{i,1} stores the Num neighbors of the ith testing instance
    for i=1:num_testing
        [temp,index]=sort(dist_matrix(i,:));
        Neighbors{i,1}=index(1:Num);
    end
    
%Computing Outputs
    Outputs=zeros(num_class,num_testing);
    for i=1:num_testing
%         if(mod(i,100)==0)
%             disp(strcat('computing outputs for instance:',num2str(i)));
%         end
        temp=zeros(1,num_class); %The number of the Num nearest neighbors of the ith instance which belong to the jth instance is stored in temp(1,j)
        neighbor_labels=[];
        for j=1:Num
            neighbor_labels=[neighbor_labels,train_target(:,Neighbors{i,1}(j))];
        end
        for j=1:num_class
            temp(1,j)=sum(neighbor_labels(j,:)==ones(1,Num));
        end
        for j=1:num_class
            Prob_in=Prior(j)*Cond(j,temp(1,j)+1);
            Prob_out=PriorN(j)*CondN(j,temp(1,j)+1);
            if(Prob_in+Prob_out==0)
                Outputs(j,i)=Prior(j);
            else
                Outputs(j,i)=Prob_in/(Prob_in+Prob_out);      
            end
        end
    end
    
%Evaluation
    Pre_Labels=zeros(num_class,num_testing);
    for i=1:num_testing
        for j=1:num_class
            if(Outputs(j,i)>=0.5)
                Pre_Labels(j,i)=1;
            else
                Pre_Labels(j,i)=-1;
            end
        end
    end
    HammingLoss=Hamming_loss(Pre_Labels,test_target);
    
    RankingLoss=Ranking_loss(Outputs,test_target);
    Coverage=coverage(Outputs,test_target);
    Average_Precision=Average_precision(Outputs,test_target);
    
    macrof1 = MacroF1(Pre_Labels,test_target);
    microf1 = MicroF1(Pre_Labels,test_target);