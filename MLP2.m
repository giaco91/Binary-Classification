clear all

%Hyperparameter
submission=0;

%read in features and target
X_tr=[];
if exist('Xv_tr.csv')==0
    Xv_tr=Feature('train',[4 4 4],'voxel');%mean,entropy,voxel
    X_tr=[X_tr Xv_tr];
    csvwrite('Xv_tr.csv',Xv_tr);
else
    Xv_tr=csvread('Xv_tr.csv');
    X_tr=[X_tr Xv_tr];
end
if  exist('Xm_tr.csv')==0
    Xm_tr=Feature('train',[1 1 1],'mean');%mean,entropy,voxel
    X_tr=[X_tr Xm_tr];
    csvwrite('Xm_tr.csv',Xv_tr);
else
    Xm_tr=csvread('Xm_tr.csv');
%     X_tr=[X_tr Xm_tr];
end
y=csvread('targets.csv');

%preprocessing
X_tr=FeatureExtr(X_tr,y,2000,'train');
X_tr=preprocess(X_tr,'train');
[X_tr,y]=Samplegenerator(X_tr,y,10);

%amount of crossvalidation
if submission==1
    cv=1;
else
    cv=28;
end
perf=ones(cv,1);
for i=1:1
%hier käme der validation forlooop
valblock=10;
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


% SVMModel = fitcsvm(Xm_tr,y);
% rng(1); % For reproducibility
% [SVMModel,ScoreParameters] = fitPosterior(SVMModel);
% [Decision,Posterior] = predict(SVMModel,Xm_tr);
% yhat=Posterior(:,2);
% Crossentropy(y,yhat)




rng('default')
layers=[200 50 20];
net=feedforwardnet(layers,'trainrp');
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;
net.trainParam.epochs = 35;
% net.trainFcn = 'trainrp';%trainrp
net.performFcn='crossentropy';
net.performParam.regularization = 0.1;
net.layers{length(layers)+1}.transferFcn = 'logsig';

%train
[net tr] = train(net,X_train',y_train');

%save
savenet=net;
save 'savemynet.mat' savenet;
%load
load savemynet;
%estimate
yhat = savenet(X_val');
% perf = crossentropy(savenet,y',yhat,{1},'regularization',0.3)
perf(i)=Crossentropy(y_val',yhat);

end
perf=sum(perf)/length(perf)



% for i=1:size(Xm_tr,2)
% figure(i)
%     for n=1:size(Xm_tr,1)
%         if y(n)==1
%             plot(Xm_tr(n,2),Xm_tr(n,i),'*r')
%             hold on
%         else
%             plot(Xm_tr(n,2),Xm_tr(n,i),'*b')
%             hold on
%         end
%     end
% hold off
% end

