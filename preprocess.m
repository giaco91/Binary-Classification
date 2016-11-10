function X_prepro= preprocess(X,train_or_test)
%normalization
if strcmp(train_or_test,'train')==1
    s2=size(X,2);
    norm=ones(s2,1);
    X_prepro=X;
    for i=1:s2
        m=max(X(:,i));
        if m>1e-6
            norm(i)=1/m;
            X_prepro(:,i)=X(:,i)*norm(i);
        else
            norm(i)=0;
            X_prepro(:,i)=X(:,i)*norm(i);
        end
    end
    csvwrite('norm.csv',norm);
else
    norm=csvread('norm.csv');
    X_prepro=X*diag(norm);
end

end

