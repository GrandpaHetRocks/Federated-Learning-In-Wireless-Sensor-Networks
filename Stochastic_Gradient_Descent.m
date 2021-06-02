tic
sig=2;
mu=2;
weight=0:0.1:10;
height=sig*randn(1,length(weight))+mu; 
% height=2*weight+3;
% height=sin(weight);
plot(weight,height,"linewidth",1.5)
slope=-5:1:5;
intercept=0:1:5;
error1=zeros(length(slope),length(intercept)); %to store mean squared errors
for i=1:length(slope)
    for j=1:length(intercept)
        predop=weight*slope(i)+intercept(j);
        error=0;
        hold on
        for k=1:length(predop)
            error=error+(height(k)-predop(k))^2;
        end
        error1(i,j)=error;
    end
end
error1;
figure
[X,Y]=meshgrid(intercept,slope);
 surf(X,Y,error1);
 xlabel("intercept")
 ylabel("slope")
 zlabel("error")
 
 %gradient descent
 niter=1000;
 steps=-inf;
 stepi=-inf;
 lrs=0.0002;  %learning rate for calculating grad wrt slope
 lri=0.0009;   %learning rate for calculating grad wrt intercept
 iter=0;
 k=slope(1);  %starting from inital value of slope
 l=intercept(1); %starting from inital value of intercept
 while(steps<=-0.01 && iter<=niter)  %terminating conditions
     gradients=0;  %gradient wrt slope
     gradienti=0;  %gradient wrt intercept
     %making a minibatch of 10 random height values (stochastic nature)
     R = randsample(100,10,false);
     R=R';
     R=sort(R);
     height1=[];
     for ii=1:10
         height1=[height1 height(R(ii))];
     end
     %height1 is our minibatch
     for i=1:length(height1)
         gradients=gradients+(-2*(height1(i)-(weight(R(i))*k+l))*weight(R(i)));
     end
     
     for i=1:length(height1)
         gradienti=gradienti+(-2*(height1(i)-(weight(R(i))*k+l)));
     end
%      gradients
     steps=gradients*lrs;
     stepi=gradienti*lri;
     k=k-steps;
     l=l-stepi;
     iter=iter+1;
 end
 predslope=k
 predintercept=l
 iter
 figure
 plot(weight,height)
 hold on
 plot(weight,weight*predslope+predintercept)
 %next 10 predicted values
 new=11:1:21;
 new*predslope+predintercept
 toc
     