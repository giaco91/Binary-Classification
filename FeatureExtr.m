function Xextr = FeatureExtr(X,y,N,method,train_or_test)
%methods:PCA,MI
 s=size(X);
if strcmp(train_or_test,'train')==1
    if N>s(2)
        N=s(2);
        Xextr=X;
    else
    if strcmp(method,'MI')==1
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
    elseif strcmp(method,'PCA')==1
        X=X';
        [U, S, V] = svd(X);
        B=U(:,1:N);
        Y=B'*X;
        Xextr=Y';
        csvwrite('PCAextr_proj.csv',B);
    elseif strcmp(method,'MaxReg')==1
        Xextr=zeros(s(1),N);
        MaxReg=zeros(s(2),1);
        for i=1:s(2)
            MaxReg(i)=score_logreg(X(:,i),y);
        end
        [sorted_MaxReg I]=sort(MaxReg);%sorted_MaxReg(1)=smallest
        HLF=num2str(sorted_MaxReg(N));
        strcat('highest loss-feature: ',HLF)
        sorted_MaxReg=sorted_MaxReg(1:N);
        I=I(1:N);
        csvwrite('MaxRegextr_idx.csv',I);
        for i=1:N
            Xextr(:,i)=X(:,I(i));
        end
    elseif strcmp(method,'PCA_MaxReg')==1
        alpha=1.5;
        scale=zeros(s(2),1);
        for i=1:s(2)
            scale(i)=score_logreg(X(:,i),y)^-alpha;
            X(:,i)=X(:,i)*scale(i);
        end
        csvwrite('scale.csv',scale);
        Xextr=FeatureExtr(X,y,N,'PCA','train'); 
    else
        error('choose a valid feature extraction method')
    end
    end
else
    if strcmp(method,'MI')==1
        I=csvread('Mutextr_idx.csv');
        for i=1:length(I)
            Xextr(:,i)=X(:,I(i));
        end
    elseif strcmp(method,'PCA')==1
        X=X';
        B=csvread('PCAextr_proj.csv');
        Y=B'*X;
        Xextr=Y';
    elseif strcmp(method,'MaxReg')==1
        I=csvread('MaxRegextr_idx.csv');
        for i=1:length(I)
            Xextr(:,i)=X(:,I(i));
        end
    elseif strcmp(method,'PCA_MaxReg')==1
        scale=csvread('scale.csv');
        B=csvread('PCAextr_proj.csv');
        X=X*diag(scale);
        Y=B'*X;
        Xextr=Y';
    else
        error('choose a valid feature extraction method')
    end
end

