clear all
close all

%Hyperparameter
submission=0;
method='DA';%NN,SVM,LR,LRlasso,DA

%read in features and target
y=csvread('targets.csv');
X_tr=[];
X_tr=[X_tr Feature('train',[4 4 4],'voxel',y)];
% X_tr=[X_tr Feature('train',[0 0 0],'mean',y)];
X_tr=[X_tr Feature('train',[4 4 4],'entropy',y)];
X_tr=[X_tr Feature('train',[1 1 1],'var',y)];
X_tr=[X_tr Feature('train',[3 3 3],'MI',y)];



% preprocessing
size(X_tr)
X_tr=FeatureExtr(X_tr,y,300,'MaxReg','train');%650
XN1=X_tr(:,1);%feature with best LR score
X_tr=preprocess(X_tr,y,'train','norm');
% X_tr=preprocess(X_tr,y,'train','shrink');
% X_tr=preprocess(X_tr,y,'train','center');

X_tr=FeatureExtr(X_tr,y,4,'PCA_MaxReg','train');
X_tr=[X_tr XN1];
% [X_tr,y]=Samplegenerator(X_tr,y,[3,1],40);


perf=classifier(X_tr,y,submission,method)



% for i=1:size(X_tr,2)
% % for i=1:1
% figure(i)
%     for n=1:size(X_tr,1)
%         if y(n)==1
%             plot(X_tr(n,1),X_tr(n,i),'*r')
%             hold on
%         else
%             plot(X_tr(n,1),X_tr(n,i),'*b')
%             hold on
%         end
%     end
% hold off
% end

