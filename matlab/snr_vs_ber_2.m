N=10000; 
snr =0:0.2:40;
data=randn(1,N)>=0; 
info = 2*data-1;
h=rand(1,1)+j*randn(1,1)
a=abs(h);
BER=qfunc((a*a*snr).^(0.5));
BER2=0.5.*(1.-((snr./(2.+snr)).^0.5));
plot(snr,BER)
hold on
plot(snr,BER2)