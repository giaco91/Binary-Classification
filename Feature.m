function X = Feature(train_or_test,N,spec,y)
     Nstring=[num2str(N(1)) num2str(N(2)) num2str(N(3))];
     dl1=11*2^N(1);%factors:2^4*11
     dl2=13*2^N(2);%factors:2^4*13
     dl3=11*2^N(3);%factors:2^4*11
     X=[];
     if strcmp(train_or_test,'train')==1
         n=278;
     else
         n=138;
     end
     if exist(strcat('X',spec,Nstring,'_',train_or_test,'.csv'))==0
         for i=1:n  
            j=num2str(i);     
            Data=load_nii(strcat(train_or_test,'_',j,'.nii'));
            Xi=Data.img;
            x=[];
            if strcmp(spec,'entropy')==1
                for k3=1:dl1-1:176-dl1+1
                    for k2=1:dl2-1:208-dl2+1
                      for k1=1:dl3-1:176-dl3+1
                          x=[x Entropy(BoxToVoxel(Xi(k1:k1+dl1-1,k2:k2+dl2-1,k3:k3+dl3-1)))];
                      end
                    end
                end
            elseif strcmp(spec,'mean')==1  
                for k3=1:dl1-1:176-dl1+1
                    for k2=1:dl2-1:208-dl2+1
                      for k1=1:dl3-1:176-dl3+1
                          x=[x mean(mean(mean(Xi(k1:k1+dl1-1,k2:k2+dl2-1,k3:k3+dl3-1),1),2),3)];
                      end
                    end
                end
                elseif strcmp(spec,'var')==1  
                    for k3=1:dl1-1:176-dl1+1
                        for k2=1:dl2-1:208-dl2+1
                          for k1=1:dl3-1:176-dl3+1
                              x=[x mean(mean(var(double(Xi(k1:k1+dl1-1,k2:k2+dl2-1,k3:k3+dl3-1))),2),3)];
                          end
                        end
                    end
            elseif strcmp(spec,'voxel')+strcmp(spec,'MI')==1
                    for k3=1:dl3-1:176-dl3+1
                        for k2=1:dl2-1:208-dl2+1
                          for k1=1:dl1-1:176-dl1+1
                              x=[x BoxToVoxel(Xi(k1:k1+dl1-1,k2:k2+dl2-1,k3:k3+dl3-1))];
                          end
                        end
                    end
            else
                error('choose a valid feature specification');
            end
            X=[X;x];  
         end
         if strcmp(spec,'MI')==1
            csvwrite(strcat('Xvoxel',Nstring,'_',train_or_test,'.csv'),X);
            size(X)
         else
          	csvwrite(strcat('X',spec,Nstring,'_',train_or_test,'.csv'),X);
         end
         if strcmp(spec,'MI')==1
             if strcmp(train_or_test,'train')==1
                 MItot=[];
                 M1=[];
                 t=0;
                 l=length(BoxToVoxel(zeros(1,1)));
                 for k3=1:dl1-1:176-dl1+1
                     for k2=1:dl2-1:208-dl2+1
                          for k1=1:dl3-1:176-dl3+1
                              Y=X(:,t*l+1:(t+1)*l);
                              m1=round(mean(Y(y==1,:),1));
                              MI=[];
                              for i=1:size(Y,1)
                                 MI=[MI;mutInfo(Y(i,:),m1)];
                              end
                              M1=[M1;m1];
                              MItot=[MItot MI];
                              t=t+1;
                          end
                      end
                 end
                X=MItot;
                csvwrite(strcat('X',spec,Nstring,'_',train_or_test,'.csv'),X);
                csvwrite('M1.csv',m1);
            else 
            MItot=[];
            M1=csvread('M1.csv');
            l=size(M1,2);
            t=0;
            for k3=1:dl1-1:176-dl1+1
                for k2=1:dl2-1:208-dl2+1
                    for k1=1:dl3-1:176-dl3+1
                        Y=X(:,t*l+1:(t+1)*l);
                        MI=[];
                        for i=1:size(Y,1)
                            MI=[MI;mutInfo(Y(i,:),M1(t+1,:))];
                        end
                        M1=[M1;m1];
                        MItot=[MItot MI];
                        t=t+1;
                     end
                 end
            end
            X=MItot;
            csvwrite(strcat('X',spec,Nstring,'_',train_or_test,'.csv'),X);
         end
     end
     else
         X=csvread(strcat('X',spec,Nstring,'_',train_or_test,'.csv'));
     end
     

end

