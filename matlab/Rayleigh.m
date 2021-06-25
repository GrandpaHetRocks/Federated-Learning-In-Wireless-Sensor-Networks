N=10000; 
snr =-20:2:40;
data=randn(1,N)>=0; %random bitstream
info = 2*data-1; 
time=0:0.0001:N*0.0001-0.0001; %10KHz
bitstream_transmit=zeros(1,N);
P=2;
for k=1:N %bpsk
    if(info(k)==1)
        bitstream_transmit(k)=-sqrt(P);
    else
        bitstream_transmit(k)=sqrt(P);
    end
end



error=[]
for i=snr %adding channel effects
    h=sqrt(P)*abs(randn(1,N)+j*randn(1,N)); %rayleigh amplitude
%     norm(h)
%     abs(h)
    snr__ = 10^(i/10);
    std = sqrt(P/snr__);
    n=(std)*randn(1,N); %AWGN
    channel=bitstream_transmit.*h+n;

    bitstream_received=zeros(1,N);
%     channel=channel./norm(h);
    for k=1:N %demodulate bpsk
        if(channel(k)>=0)  
            bitstream_received(k)=0;
        else
            bitstream_received(k)=1;
        end
    end

    hold on
%     plot(time,bitstream_received)


    ber=sum(xor(data,bitstream_received)/N);
    error=[error ber];
end

semilogy(snr,error)
grid on
error