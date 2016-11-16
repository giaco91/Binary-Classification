function perf = classifier(X_tr,y,submission,method)
    %amount of crossvalidation
    if submission==1
        cv=1;
    else
        cv=55;
    end
    perf=[];
    t=[0 1];
    for i=1:cv %the validation loop
        valblock=5;
        valbegin=round((size(X_tr,1)-valblock)/cv*i);
        val_set=[valbegin valbegin+valblock];
        if submission==1
            train_set=[1 2 3 size(X_tr,1)];
        else
            train_set=[1 val_set(1)-1 val_set(2)+1 size(X_tr,1)]; %prozentual oder zeilenvektor (max 278)
        end
        y_train=[y(train_set(1):train_set(2),:);y(train_set(3):train_set(4),:)];
        X_train=[X_tr(train_set(1):train_set(2),:);X_tr(train_set(3):train_set(4),:)];
        if submission==0
            y_val=y(val_set(1):val_set(2),:);
            X_val=X_tr(val_set(1):val_set(2),:);
        end


        %LR------
        if strcmp(method,'LR')==1
            [b,dev,stats] = glmfit(X_train,y_train,'binomial','link','logit');
            yhat = glmval(b,X_val,'logit');
    %         for i=1:length(yhat)
    %             if y_val(i)==0
    %                 if t(1)<yhat(i)
    %                     t(1)=yhat(i);
    %                 end
    %             else
    %                 if t(2)>yhat(i)
    %                     t(2)=yhat(i);
    %                 end
    %             end
    %         end

            %risk
    %         for i=1:length(yhat)
    %             if yhat(i)>0.95
    %                 yhat(i)=1-1e-6;
    %             elseif yhat(i)<0.07
    %                 yhat(i)=1e-6;
    %             end
    %         end


            perf=[perf Crossentropy(y_val,yhat)];

        %LRlasso------
        elseif strcmp(method,'LRlasso')==1
            %find right lambda:
    %         [B,FitInfo] = lassoglm(X_train,y_train,'binomial','NumLambda',25,'CV',10);
    %         lambda=FitInfo.Lambda1SE
            lambda=0.03;
            [B,FitInfo] = lassoglm(X_train,y_train,'binomial','Lambda',lambda);
            b=[FitInfo.Intercept;B];
            yhat = glmval(b,X_val,'logit');

            perf=[perf Crossentropy(y_val,yhat)];


        %  SVM----------
        elseif strcmp(method,'SVM')==1
            SVMModel = fitcsvm(X_train,y_train);
            rng(1); % For reproducibility
            [SVMModel,ScoreParameters] = fitPosterior(SVMModel);
            [Decision,Posterior] = predict(SVMModel,X_val);
            yhat=Posterior(:,2);
            perf=[perf Crossentropy(y_val,yhat)];



    %         % Neural Netowrk---------------
        elseif strcmp(method,'NN')==1
            rng(2)
            layers=[8];
            net=feedforwardnet(layers,'trainrp');
            net.divideParam.trainRatio = 100/100;
            net.divideParam.valRatio = 0/100;
            net.divideParam.testRatio = 0/100;
            net.trainParam.epochs = 12;
            % net.trainFcn = 'trainrp';%trainrp
            net.performFcn='crossentropy';
            net.performParam.regularization = 0.1;
            net.layers{length(layers)+1}.transferFcn = 'logsig';        
            [net tr] = train(net,X_train',y_train');%train     
            savenet=net;%save
            % save 'savemynet.mat' savenet;       
            % load savemynet;%load
            yhat = savenet(X_val')';%estimate
            perf=[perf Crossentropy(y_val,yhat)];
        else
            error('choose a valid method')
        end
    end
    perf=sum(perf)/length(perf);
    % t


end

