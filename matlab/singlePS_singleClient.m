N=10000; 
snr =-20:2:40;
data=randn(1,N)>=0; 
info = 2*data-1;
time=0:1:N*1-1;
bitstream_transmit=zeros(1,N);
P=2;
for k=1:N %bpsk
    if(info(k)==1)
        bitstream_transmit(k)=-sqrt(P);
    else
        bitstream_transmit(k)=sqrt(P);
    end
end

%plot(time,data)

error=[]
for i=snr
    h=sqrt(P)*abs(randn(1,N)+j*randn(1,N));
%     norm(h)
%     abs(h)
    snr__ = 10^(i/10);
    std = sqrt(P/snr__);
    n=(std)*randn(1,N);
    channel=bitstream_transmit.*h+n;

    bitstream_received=zeros(1,N);
%     channel=channel./norm(h);
    for k=1:N
        if(channel(k)>=0)  %demodulate bpsk
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