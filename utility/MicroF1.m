function microf1 = MicroF1(Pre_Labels,test_target)
%Computing the Macro_AUC
%Pre_Labels       - If the ith testing instance belongs to the jth class, then Pre_Labels(j,i) is +1, otherwise Pre_Labels(j,i) is -1
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

    [num_class,num_instance]=size(Pre_Labels);
	num_P_instance = zeros(num_class,1);%number of positive instance for every label
    num_N_instance = zeros(num_class,1);
	count_valid_label = 0;
    fm = zeros(num_class,1);
    sumTP=0;
    sumTN=0;
    sumFP=0;
    sumFN=0;
    for i = 1:num_class
	num_P_instance(i,1) = sum(test_target(i,:) == 1);
    num_N_instance(i,1) = num_instance - num_P_instance(i,1);
	num_P = sum(Pre_Labels(i,:) == 1);
	num_N = num_instance - num_P;
	pre=Pre_Labels(i,:);
	pre0=pre;
	instance=test_target(i,:);
	pre0(pre0==-1)=0;
	TP=sum(pre0==instance);
	pre0=pre;
	pre0(pre0==1)=0;
	TN=sum(pre0==instance);
	FP=num_P-TP;
	FN=num_N-TN;
    sumTP=sumTP+TP;
    sumTN=sumTN+TN;
    sumFP=sumFP+FP;
    sumFN=sumFN+FN;

    end
    

	microf1=2*sumTP/(2*sumTP+sumFN+sumFP);



end

