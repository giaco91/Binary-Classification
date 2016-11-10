function X = Feature(train_or_test,N,spec)
     dl1=11*2^N(1);%factors:2^4*11
     dl2=13*2^N(2);%factors:2^4*13
     dl3=11*2^N(3);%factors:2^4*11
     X=[];
     if strcmp(train_or_test,'train')==1
         n=278;
     else
         n=138;
     end   
     for i=1:n  
        j=num2str(i);     
        Data=load_nii(strcat(train_or_test,'_',j,'.nii'));
        Xi=Data.img;
        x=[];
        I=0;
        if strcmp(spec,'entropy')==1
            for k3=1:dl1-1:176-dl1+1
                for k2=1:dl2-1:208-dl2+1
                  for k1=1:dl3-1:176-dl3+1
                      I=I+1;
                      x=[x Entropy(BoxToVoxel(Xi(k1:k1+dl1-1,k2:k2+dl2-1,k3:k3+dl3-1)))];
                  end
                end
            end
        elseif strcmp(spec,'mean')==1  
            for k3=1:dl1-1:176-dl1+1
                for k2=1:dl2-1:208-dl2+1
                  for k1=1:dl3-1:176-dl3+1
                      I=I+1;
                      x=[x mean(mean(mean(Xi(k1:k1+dl1-1,k2:k2+dl2-1,k3:k3+dl3-1),1),2),3)];
                  end
                end
            end
        else    
            for k3=1:dl3-1:176-dl3+1
                for k2=1:dl2-1:208-dl2+1
                  for k1=1:dl1-1:176-dl1+1
                      I=I+1;
                      x=[x BoxToVoxel(Xi(k1:k1+dl1-1,k2:k2+dl2-1,k3:k3+dl3-1))];
                  end
                end
            end
        end
        X=[X;x];
     end
end

