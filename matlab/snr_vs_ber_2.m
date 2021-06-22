N=10000; 
snr =-20:2:40;
snr=10.^(snr/10);
h=rand(1,1)+j*randn(1,1)
a=abs(h);
BER=qfunc((a*a*snr).^(0.5));
BER2=0.5.*(1.-((snr./(2.+snr)).^0.5));
plot(snr,BER)
hold on
plot(snr,BER2)