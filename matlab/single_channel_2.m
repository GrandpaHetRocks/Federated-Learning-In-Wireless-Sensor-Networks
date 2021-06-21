N=10000; 
snr =0:0.2:40;
data=randn(1,N)>=0; 
info = 2*data-1;
a=5;
theta=30;
h=a*exp(j*theta);
BER=qfunc((a*a*snr).^(0.5))
BER2=0.5.*(1.-((snr./(2.+snr)).^0.5))
plot(snr,BER)
hold on
plot(snr,BER2)