%federated SGD with number of clients=3, C=1

tic
sig=2;
mu=2;
weight=0:1:10;
slope=0:1:5;
intercept=20:1:25;
% height1=sig*randn(1,length(weight))+mu;
% height2=sig*randn(1,length(weight))+mu; 
% height3=sig*randn(1,length(weight))+mu; 
height1=(25-20).*rand(1,length(weight)) + 20;
height2=(25-20).*rand(1,length(weight)) + 20;
height3=(25-20).*rand(1,length(weight)) + 20;
hold on
plot(weight,height1,"linewidth",1.5)
plot(weight,height2,"linewidth",1.5)
plot(weight,height3,"linewidth",1.5)
error1=zeros(length(slope),length(intercept)); %to store mean squared errors
error2=zeros(length(slope),length(intercept)); %to store mean squared errors
error3=zeros(length(slope),length(intercept)); %to store mean squared errors

%calculating mean squared errors
for i=1:length(slope)
    for j=1:length(intercept)
        predop=weight*slope(i)+intercept(j);
        error=0;
        hold on
        plot(weight,predop)
        for k=1:length(predop)
            error=error+(height1(k)-predop(k))^2;
        end
        error1(i,j)=error;
    end
end
for i=1:length(slope)
    for j=1:length(intercept)
        predop=weight*slope(i)+intercept(j);
        error=0;
        hold on
        plot(weight,predop)
        for k=1:length(predop)
            error=error+(height2(k)-predop(k))^2;
        end
        error2(i,j)=error;
    end
end
for i=1:length(slope)
    for j=1:length(intercept)
        predop=weight*slope(i)+intercept(j);
        error=0;
        hold on
        plot(weight,predop)
        for k=1:length(predop)
            error=error+(height3(k)-predop(k))^2;
        end
        error3(i,j)=error;
    end
end
%plotting mean squared errors
figure
[X,Y]=meshgrid(intercept,slope);
 surf(X,Y,error1);
 xlabel("intercept")
 ylabel("slope")
 zlabel("error")
 figure
[X,Y]=meshgrid(intercept,slope);
 surf(X,Y,error2);
 xlabel("intercept")
 ylabel("slope")
 zlabel("error")
 figure
[X,Y]=meshgrid(intercept,slope);
 surf(X,Y,error3);
 xlabel("intercept")
 ylabel("slope")
 zlabel("error")
 
%fed SGD

niter=1000;
 steps=-inf;
 stepi=-inf;
 lrs=0.0002;  %learning rate for calculating grad wrt slope
 lri=0.01;   %learning rate for calculating grad wrt intercept
 iter=0;
 k=slope(1);  %starting from inital value of slope
 l=intercept(1); %starting from inital value of intercept
 while(steps<=-0.01 && iter<=niter)  %terminating conditions
     gradients1=0;  %gradient wrt slope (client 1)
     gradienti1=0;  %gradient wrt intercept (client 1)
     gradients2=0;  %gradient wrt slope (client 2)
     gradienti2=0;  %gradient wrt intercept (client 2) 
     gradients3=0;  %gradient wrt slope (client 3)
     gradienti3=0;  %gradient wrt intercept (client 3)
     for i=1:length(weight)
         gradients1=gradients1+(-2*(height1(i)-(weight(i)*k+l))*weight(i));
     end
     
     for i=1:length(weight)
         gradienti1=gradienti1+(-2*(height1(i)-(weight(i)*k+l)));
     end
     
     for i=1:length(weight)
         gradients2=gradients2+(-2*(height2(i)-(weight(i)*k+l))*weight(i));
     end
     
     for i=1:length(weight)
         gradienti2=gradienti2+(-2*(height2(i)-(weight(i)*k+l)));
     end
     
     for i=1:length(weight)
         gradients3=gradients3+(-2*(height3(i)-(weight(i)*k+l))*weight(i));
     end
     
     for i=1:length(weight)
         gradienti3=gradienti3+(-2*(height3(i)-(weight(i)*k+l)));
     end
%      gradients
     steps=(gradients1+gradients2+gradients3)*lrs/3;  %server side averaging
     stepi=(gradienti1+gradienti2+gradienti3)*lri/3;  %server side averaging
     k=k-steps;
     l=l-stepi;
     iter=iter+1;
 end
  predslope=k
 predintercept=l
 iter
 figure
 plot(weight,height1)
 hold on
 plot(weight,height2)
 plot(weight,height3)
 plot(weight,weight*predslope+predintercept,"linewidth",1.5)
 toc
 
 %next 10 temperature predictions
    new=11:1:21;
 predictions=new*predslope+predintercept