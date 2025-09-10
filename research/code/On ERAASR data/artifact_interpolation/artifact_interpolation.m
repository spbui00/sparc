clear
clear variables; 
clc
% Nin = par.ain*sin(2*pi*par.fin*t)';

% load('+data/50ad9_paramSweep4.mat') % DBS data set
% fsData = fs_data;
% tEpoch = t_epoch;
% %xlims = [-200 600];
% chanIntList = [5,6,7,9,10];
% trainDuration = [0 500];
% minDuration = 0.250; % minimum duration of artifact in ms
% Ain = 4*dataInt(:,1,1); % needed to be multiplied by 4 from raw recordin

dat=load('../exampleDataTensor.mat');
data_in=dat.data_trials_by_time_by_channels;
Ain_t=data_in(1,:,1);
Ain_t2=data_in(1,:,2);
Ain_nt = -1 + 2.*(Ain_t - min(Ain_t))./(max(Ain_t) - min(Ain_t));
Ain_nt2 = -1 + 2.*(Ain_t2 - min(Ain_t2))./(max(Ain_t2) - min(Ain_t2));

ups = 1;
dns = 1;
Ain = resample(Ain_nt2',ups,dns);

% close all;
% Full input signal swing with 1-bit MSB
sigma                   = 0.1;         % Mismatch sigma 
bit                     = 9;           % SAR resolution
Ci                      = 2.^(0:bit-1);
% Input signal
bin                     = 225;            % Number of signal bin
Order_fft               = 18;
N_fft                   = 2^Order_fft;          % Number of conversions
Num                     = N_fft+1000;          % Number of conversions
f                       = 1/Num*bin;   % Signal frequency
A                       = 0.8;          % Signal amplitude
Vsin                    = sin(2*pi*f*(0:Num-1)); % Sine input
Vin                     = zeros(1,N_fft);
for i = 1:Num
    Vin(i) = Ain(mod(i-1, length(Ain)) + 1);
end
Vin=A*Vin/max(abs(Vin));
LSB                     = 1/2^(bit-1);
gain                    = 4;
Dsar                    = zeros(1,N_fft);                             
Dsarp                   = zeros(1,N_fft);                            
Dsarn                   = zeros(1,N_fft);                                                       
Diadc                   = zeros(1,N_fft);                            
Dout                    = zeros(1,N_fft);
Vint                    = zeros(1,N_fft);
Vint1                   = zeros(1,N_fft);
Vsamp                   = zeros(1,N_fft);
Vres                    = zeros(1,N_fft);
thermal                 = rand/(2^bit)*0.02;
% C                       = zeros(1,bit);   % Capacitor array
               D        = zeros(1,bit);     % SAR digital output


%%%%%%%% Define the DAC array %%%%%%%%%%%%
    Cu                  = 1 + sigma * randn(1,2^bit);                       % Add mismatch to unit capacitor
%     load('Cu.mat');
    for k               = 1:1:bit
        C(k)            = sum(Cu(2^(k-1):2^k-1));                           % Allot unit capacitors to every bit  
    end 
    Ctot                = C(bit)*2; 



%%%%%%%%%%%%% SAR Operation %%%%%%%%%%%%%
% first conversion
        Vsamp(1)=0;
        Vint1(1)=Vin(1)-Dout(1);
        Vres(1)            = Vsamp(1);               % Sample
        Vint(1)         =0;
        Vs(1)           = Vres(1)+4*Vint(1);
        Dsar(1)         = 0;                     % Final SAR output


        for j           = bit:-1:1
            if Vs(1)     > 0
                D(j)    = 1;                    % D1 is SAR ADC digital output array
            else
                D(j)    = -1;
            end
            Vres(1)     = Vres(1) - D(j) * C(j)/Ctot;        % SAR residue
            Dsar(1)     = Dsar(1) + D(j) * 2^(j-bit-1); 
        end
        Dlsb(1)         = Dsar(1) - D(bit)/2;
        Vlsb(1)         = sum(D(1:bit-1).*C(1:bit-1))/Ctot;         
        Dout(1)         = Dsar(1);   

    for i               = 2:length(Vin)       
 %%%%%%  1-bit prediction    %%%%%%%%%%%%%   
        vth             = 0.25;
        if Dout(i-1)      > vth
            Dp(i)     = 1;
        elseif Dout(i-1)  < -vth
            Dp(i)     = -1;
        else
            Dp(i)     = 0;
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        Vres(i)         = (Vlsb(i-1)+Vin(i)+Vres(i-1) - Dp(i)*C(bit)/Ctot+rand/(2^bit)*0.02/gain);
        if mod(i-1000, 1024) == 0
        Vint(i)=0;
        else
        Vint(i)         = 0.8*Vint(i-1)+0.2*Vres(i-1)+rand/(2^bit)*0.02/gain;
        end
        Dsar(i)         = 0;                     % Final SAR output
        
        for j           = bit:-1:1
            if gain*Vres(i)+4*Vint(i)  > 0
                D(j)    = 1;                    % D1 is SAR ADC digital output array
            else
                D(j)    = -1;
            end
            Vres(i)     = Vres(i) - D(j) * C(j)/Ctot;        % SAR residue
            Dsar(i)     = Dsar(i) + D(j) * 2^(j-bit-1); 
        end

        Dlsb(i)         = Dsar(i) - D(bit)/2;
        Vlsb(i)         = sum(D.*C)/Ctot - D(bit)*C(bit)/Ctot; 
        Dout(i)         = Dsar(i) - Dlsb(i-1) + Dp(i)/2; % Dout = Dsar - Dlsb*z^(-1) + Dp/2
        Vs(i)           = Vin(i)-Dout(i);
    end
 %%%%%%  scalling    %%%%%%%%%%%%%
 % for i = 1:length(Dout)
 %    Dout(i)=log2(abs(Dout(i)));
 % end
 % Dout=Dout/8;
 %%%%%%  Intepolation    %%%%%%%%%%%%%
   pthreshold    =0.05;
   nthreshold    =-0.05;
   flag_flat                = zeros(1,N_fft);
   flag_1                   = zeros(1,N_fft);
   Dtemp_1                  = zeros(1,N_fft);
   Dtemp_2                  = zeros(1,N_fft);
   for i = 2:length(Dout)-10000
   % if -threshold<Dout(i)-Dout(i-1)<threshold
   %    Dinte(i)=Dout(i);
   if Dout(i)-Dout(i-1) < nthreshold
      flag_1(i)=1;
      if flag_flat(i-1)==1 
          if Dout(i-1)<0.2 && Dout(i-1)>-0.2
          Dtemp_1(i)=Vin(i-1);
          flag_flat(i)=0;
          else 
              Dtemp_1(i)=0;
          end
      end
   end
   if Dout(i)-Dout(i-1) < pthreshold && Dout(i)-Dout(i-1) > nthreshold
      flag_flat(i)=1;
        if flag_1(i-1)==1 
            if Dout(i)<0.2 && Dout(i)>-0.2
           Dtemp_2(i)=Vin(i);
           flag_1(i)=0;
            else
                Dtemp_2(i)=0;
            end
        end
   end
   end
 % Find indices of zero values
zeroIndices = find(Dtemp_1 == 0);
zeroIndices_2 = find(Dtemp_2 == 0);
% Find indices of non-zero values
nonZeroIndices = find(Dtemp_1 ~= 0);
nonZeroIndices_2 = find(Dtemp_2 ~= 0);
% Perform linear interpolation at zero indices
Dtemp_1(zeroIndices) = interp1(nonZeroIndices, Dtemp_1(nonZeroIndices), zeroIndices, 'linear');
Dtemp_2(zeroIndices_2) = interp1(nonZeroIndices_2, Dtemp_2(nonZeroIndices_2), zeroIndices_2, 'linear');
% Display the result
 % disp(Dtemp_1);
 plot(Dout);
 hold on
 % plot(Vin);
 % hold on
 plot (Dtemp_1);
 hold on
 plot (Dtemp_2);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Perform FFT
% win             = hodie(length(Dout));
% Dout            = win'.*Dout;
% Dout            = Dout - mean(Dout);
% FFT             = fft(Dout(100:N_fft+99),N_fft);
% Dout_FFT        = abs(fft(Dout,N_fft));
% Dout_FFT_db     = 20*log10(Dout_FFT);
% Dout_FFT_power  = Dout_FFT.^2;  
% [bp bp]         = max(Dout_FFT_db(1:2*bin));
% peak            = Dout_FFT_db(bin);
% signal_power    = sum(Dout_FFT_power(bin-6:bin+6));
% total_power     = sum(Dout_FFT_power(1:floor(N_fft/2)));
% SNDR            = 10*log10(signal_power/(total_power-signal_power));
% ENOB            = (SNDR-1.76)/6.02; % ENOB from SNDR 
% 
% % Plot SNDR vs OSR
% figure(1);
% OSR         = 1:1:128;
% SNDR_tab    = zeros(1,length(OSR));
% for k = 1:length(OSR)
%     SNDR_tab(k) = 10*log10(signal_power/(sum(Dout_FFT_power(1:floor(N_fft/2/k)))-signal_power));
% end
% % 
% semilogx(OSR,SNDR_tab,'g','LineWidth',2);
% set(gca,'FontSize',12,'color','none');
% xlabel('OSR');
% ylabel('SNDR');
% axis tight;
% grid on;
% hold on;
% f = gcf;
% f.Position = [300 300 600 360];
% %legend('w/o prediction');
% legend('w/ prediction');
% set(legend,'FontSize',12,'EdgeColor',[0.94 0.94 0.94]);
% % 
% figure(2);
% x_axis = (1:N_fft/2)/N_fft;
% semilogx(x_axis, Dout_FFT_db(1:N_fft/2) - peak + 20*log10(A),'b','LineWidth',2);
% set(gca,'FontSize',12,'color','none');
% xlim([0.0001, 1/2]);
% ylim([-140, 0]);
% xlabel('Normalized Frequency');
% ylabel('Spectrum (dBFS)');
% set(gca,'xtick',[10^(-3) 10^(-2) 10^(-1) 2^(-1)]);
% % legend('After MES');
% legend('Random selection (DEM)');
% % legend('w/ prediction');
% set(legend,'FontSize',12,'EdgeColor',[0.94 0.94 0.94]);
% grid on;
% hold on;
% f = gcf;
% f.Position = [300 300 600 360];