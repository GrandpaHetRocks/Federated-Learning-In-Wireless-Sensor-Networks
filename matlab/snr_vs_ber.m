N=10000; 
snr =-20:2:40;
data=randn(1,N)>=0; 
info = 2*data-1;
M=2; 
Rm=log2(M); 
Rc=1; 
BER2 = zeros(1,length(snr)); 
index2=1;

BER3 = zeros(1,length(snr)); 
index3=1;
index4=1;
BER4=zeros(1,length(snr));

for k2=snr

EbN0 = 10.^(k2/10); 
noiseSigma = sqrt(1./(2*EbN0)); 
n1 = noiseSigma*randn(1,length(info));
n2 = noiseSigma*randn(1,length(info));
h1 = sqrt(0.5)*abs(randn(1,N) + j*randn(1,N)); %rayleigh amplitude 1
h2 = sqrt(0.5)*abs(randn(1,N) + j*randn(1,N)); %rayleigh amplitude 1
rayleigh=[h1 ; h2];
noise=[n1; n2];
coeff=1/(norm(rayleigh));
weights=coeff*[rayleigh];
y_1 = info.*(h1)+(n1);
y_2=info.*(h2)+n2;
y=[y_1;y_2];
y_maximal=sum(y.*((weights)));
estimated_bits=[y_maximal>=0];
error=[sum(xor(data,estimated_bits))/(length(data))];

%xor(data,estimated_bits(1,:))

%received = info + noise;
%estimatedBits=(y_maximal[1]>=0;y_maximal[2]>=0)


BER2(index2) =0.5*(error+error) ;
index2=index2+1;
end

for k3=snr

EbN0 = 10.^(k3/10); 
noiseSigma = sqrt(1./(2*EbN0)); 
n1 = noiseSigma*randn(1,length(info));
n2 = noiseSigma*randn(1,length(info));
n3 = noiseSigma*randn(1,length(info));
h1 = sqrt(0.5)*abs(randn(1,N) + j*randn(1,N)); %rayleigh amplitude 1
h2 = sqrt(0.5)*abs(randn(1,N) + j*randn(1,N)); %rayleigh amplitude 1
h3 = sqrt(0.5)*abs(randn(1,N) + j*randn(1,N));
rayleigh=[h1 ; h2;h3];
noise=[n1; n2;n3];
coeff=1/(norm(rayleigh));
weights=coeff*[rayleigh];
y_1 = info.*(h1)+(n1);
y_2=info.*(h2)+n2;
y_3=info.*(h3)+n3;
y=[y_1;y_2;y_3];
y_maximal=sum(y.*((weights)));
estimated_bits=[y_maximal>=0];
error=[sum(xor(data,estimated_bits(1,:)))/(length(data))];
%xor(data,estimated_bits(1,:))

%received = info + noise;
%estimatedBits=(y_maximal[1]>=0;y_maximal[2]>=0)


BER3(index3) =(error) ;
index3=index3+1;
end
for k3=snr
EbN0 = 10.^(k3/10); 
noiseSigma = sqrt(1./(2*EbN0)); 
n1 = noiseSigma*randn(1,length(info));
n2 = noiseSigma*randn(1,length(info));
n3 = noiseSigma*randn(1,length(info));
n4 = noiseSigma*randn(1,length(info));
h1 = sqrt(0.5)*abs(randn(1,N) + j*randn(1,N)); 
h2 = sqrt(0.5)*abs(randn(1,N) + j*randn(1,N)); 
h3 = sqrt(0.5)*abs(randn(1,N) + j*randn(1,N));
h4 = sqrt(0.5)*abs(randn(1,N) + j*randn(1,N));
rayleigh3=[h1 ; h2;h3;h4];
noise=[n1; n2;n3;n4];
coeff=1/(norm(rayleigh3));
weights=coeff*[rayleigh3];
y_1 = info.*(h1)+(n1);
y_2=info.*(h2)+n2;
y_3=info.*(h3)+n3;
y_4=info.*(h4)+n4;
y=[y_1;y_2;y_3;y_4];
y_maximal=sum(y.*((weights)));
estimated_bits=[y_maximal>=0];
error=[sum(xor(data,estimated_bits))/(length(data))];

%xor(data,estimated_bits(1,:))

%received = info + noise;
%estimatedBits=(y_maximal[1]>=0;y_maximal[2]>=0)


BER4(index4) =error ;
index4=index4+1;
end

semilogy(snr,(BER2),'r-');
xlim([-20 40])
hold on
semilogy(snr,(BER3),'b-');
xlim([-20 40])
hold on
semilogy(snr,(BER4),'g-');
legend('mrc2','mrc3','mrc4')

grid on