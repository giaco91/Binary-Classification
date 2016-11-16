function score = score_logreg(x,y)
cv=15;
perf=[];
if length(unique(x))/length(x)<0.2 %outsort zero-features
    score=1;  %big enough
else
    for i=1:cv %the validation loop
        %set up validation and training set
        valblock=15;
        valbegin=round((length(x)-valblock)/cv*i);
        val_set=[valbegin valbegin+valblock];
        train_set=[1 val_set(1)-1 val_set(2)+1 length(x)]; %prozentual oder zeilenvektor (max 278)
        y_train=[y(train_set(1):train_set(2));y(train_set(3):train_set(4))];
        x_train=[x(train_set(1):train_set(2));x(train_set(3):train_set(4))];
        y_val=y(val_set(1):val_set(2));
        x_val=x(val_set(1):val_set(2));

        %regression
        [b,dev,stats] = glmfit(x_train,y_train,'binomial','link','logit');
        yhat = glmval(b,x_val,'logit');
        perf=[perf Crossentropy(y_val,yhat)];
    end
    score=sum(perf)/length(perf);
end
