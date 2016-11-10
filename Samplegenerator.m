function [Xgen ygen] = Samplegenerator(X,y,factor)
    %X, feature-data matrix with size(X,1)=#samples and size(X,2)=#features
    %y, binary classification vector with classes {0,1} and length(y)=size(X,1)
    %factor, the factor of generated data, i.e. size(Xgen,1)/size(X,1)=factor

    s=size(X);
    n=[length(y)-sum(y) sum(y)];
    X0=zeros(n(1),s(2));%null-class features
    X1=zeros(n(2),s(2));%one-class features
    sigma=zeros(s(1),1);%estimated sigmas
    [nmax Imax]=max(n);
    Imin=mod(Imax,2)+1;
    nmin=n(Imin);
    k0=1;
    k1=1;
    for i=1:s(1)
        if y(i)==0
            X0(k0,:)=X(i,:);
            k0=k0+1;
        else
            X1(k1,:)=X(i,:);
            k1=k1+1;
        end
    end

    X01={X0 X1};%the two class-features
    X=[X0;X1];%the sorted features
    for i=1:nmax   
            %shortest distance to a sample of the other class
            dmin=min(sum(abs(repmat(X01{Imax}(i,:),nmin,1)-X01{Imin}),2));
            sigma(i+(Imax-1)*n(1))=dmin/2; %standarddeviation = dmin/2 (model)  
    end
    for i=1:nmin 
     
            dmin=min(sum(abs(repmat(X01{Imin}(i,:),nmax,1)-X01{Imax}),2));
            sigma(i+(Imin-1)*n(1))=dmin/4; 
    end
    Xgen=zeros(factor*s(1),s(2));
    ygen=zeros(factor*s(1),1);
    Xgen(1:s(1),1:s(2))=X;
    ygen(1:s(1))=y;
    k=1;
    for i=1:factor-1
        for j=1:n(1)
            Xgen(s(1)+k,:)=normrnd(X(j,:),sigma(j));
            ygen(s(1)+k)=0;
            k=k+1;
        end
        for j=n(1)+1:s(1)
            Xgen(s(1)+k,:)=normrnd(X(j,:),sigma(j));
            ygen(s(1)+k)=1;
            k=k+1;
        end
    end
    %randomize sampleorder
    idx=randperm(factor*s(1));
    Xgen=Xgen(idx,:);
    ygen=ygen(idx);
end

