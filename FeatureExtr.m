function Xextr = FeatureExtr(X,y,N,train_or_test)
 s=size(X);
if strcmp(train_or_test,'train')==1
    if N>s(2)
        N=s(2);
        Xextr=X;
    else
        Xextr=zeros(s(1),N);
        mutInf=zeros(s(2),1);
        for i=1:s(2)
            mutInf(i)=mutInfo(round(X(:,i)),round(y));
        end
        [sorted_mutInf I]=sort(mutInf);%sorted_mutInf(1)=smallest
        I=I(end-N+1:end);
        csvwrite('Mutextr_idx.csv',I);
        for i=1:N
            Xextr(:,i)=X(:,I(i));
        end   
    end
else
    I=csvread('Mutextr_idx.csv');
    for i=1:length(I)
        Xextr(:,i)=X(:,I(i));
    end
end

