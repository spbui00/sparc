clear
clear variables; 
clc
close all;

% % Plot corelation among channels
% dat = load('exampleDataTensor.mat');
% data_in = dat.data_trials_by_time_by_channels;
% 
% data_in = data_in(:, 1:1000, :); % no artifact
% % Reshape data to (total_trials * num_timesteps) x total_channels
% total_trials = size(data_in, 1);
% num_timesteps = size(data_in, 2);
% total_channels = size(data_in, 3);
% 
% reshaped_data = reshape(data_in, total_trials * num_timesteps, total_channels);
% 
% % Compute correlation matrix
% corr_matrix = corr(reshaped_data);
% 
% % Plot the correlation matrix
% figure;
% imagesc(corr_matrix);
% colorbar;
% colormap(jet);
% caxis([-1, 1]); % Set color scale to correlation range (-1 to 1)
% axis square;
% title('Channel Correlation Matrix');
% xlabel('Channels');
% ylabel('Channels');
% 
% % Improve visibility
% set(gca, 'FontSize', 12);





% load original data
dat=load('exampleDataTensor.mat');
data_in=dat.data_trials_by_time_by_channels;
total_trials = size(data_in,1);
total_channels = size(data_in,3);
num_timesteps = size(data_in,2);

sampling_rate = 30000; % 30 kHz
time_in_ms = (0:size(data_in, 2)-1) / sampling_rate * 1000; % Convert to milliseconds
%% ADC
% Ain_t=data_in(1,:,1);
% Ain_t = -1 + 2.*(Ain_t - min(Ain_t))./(max(Ain_t) - min(Ain_t)); % for ADC input, normalize to [-1,1]
% Ain = Ain_t';
% % Full input signal swing with 1-bit MSB
% sigma                   = 0.1;         % Mismatch sigma 
% bit                     = 9;           % SAR resolution
% Ci                      = 2.^(0:bit-1);
% % Input signal
% bin                     = 225;            % Number of signal bin
% Order_fft               = 18;
% N_fft                   = 2^Order_fft;          % Number of conversions
% Num                     = N_fft+1000;          % Number of conversions
% f                       = 1/Num*bin;   % Signal frequency
% A                       = 0.8;          % Signal amplitude
% Vsin                    = sin(2*pi*f*(0:Num-1)); % Sine input
% Vin                     = zeros(1,N_fft);
% for i = 1:Num
%     Vin(i) = Ain(mod(i-1, length(Ain)) + 1);
% end
% Vin=A*Vin/max(abs(Vin)); % normalized data
% LSB                     = 1/2^(bit-1);
% gain                    = 100;
% Dsar                    = zeros(1,N_fft);                             
% Dsarp                   = zeros(1,N_fft);                            
% Dsarn                   = zeros(1,N_fft);                                                       
% Diadc                   = zeros(1,N_fft);                            
% Dout                    = zeros(1,N_fft);% after ADC
% Vint                    = zeros(1,N_fft); 
% Vint1                   = zeros(1,N_fft);
% Vsamp                   = zeros(1,N_fft);
% Vres                    = zeros(1,N_fft);
% thermal                 = rand/(2^bit)*0.02;
% % C                       = zeros(1,bit);   % Capacitor array
%                D        = zeros(1,bit);     % SAR digital output
% 
% 
% %%%%%%%% Define the DAC array %%%%%%%%%%%%
%     Cu                  = 1 + sigma * randn(1,2^bit);                       % Add mismatch to unit capacitor
% %     load('Cu.mat');
%     for k               = 1:1:bit
%         C(k)            = sum(Cu(2^(k-1):2^k-1));                           % Allot unit capacitors to every bit  
%     end 
%     Ctot                = C(bit)*2; 
% 
% 
% 
% %%%%%%%%%%%%% SAR Operation %%%%%%%%%%%%%
% % first conversion
%         Vsamp(1)=0;
%         Vint1(1)=Vin(1)-Dout(1);
%         Vres(1)            = Vsamp(1);               % Sample
%         Vint(1)         =0;
%         Vs(1)           = Vres(1)+4*Vint(1);
%         Dsar(1)         = 0;                     % Final SAR output
% 
% 
%         for j           = bit:-1:1
%             if Vs(1)     > 0
%                 D(j)    = 1;                    % D1 is SAR ADC digital output array
%             else
%                 D(j)    = -1;
%             end
%             Vres(1)     = Vres(1) - D(j) * C(j)/Ctot;        % SAR residue
%             Dsar(1)     = Dsar(1) + D(j) * 2^(j-bit-1); 
%         end
%         Dlsb(1)         = Dsar(1) - D(bit)/2;
%         Vlsb(1)         = sum(D(1:bit-1).*C(1:bit-1))/Ctot;         
%         Dout(1)         = Dsar(1);   
% 
%     for i               = 2:length(Vin)       
%  %%%%%%  1-bit prediction    %%%%%%%%%%%%%   
%         vth             = 0.25;
%         if Dout(i-1)      > vth
%             Dp(i)     = 1;
%         elseif Dout(i-1)  < -vth
%             Dp(i)     = -1;
%         else
%             Dp(i)     = 0;
%         end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
% % model thermal noise of ADC
%         Vres(i)         = (Vlsb(i-1)+Vin(i)+Vres(i-1) - Dp(i)*C(bit)/Ctot+rand/(2^bit)*0.02/gain);
%         if mod(i-1000, 1024) == 0
%         Vint(i)=0;
%         else
%         Vint(i)         = 0.8*Vint(i-1)+0.2*Vres(i-1)+rand/(2^bit)*0.02/gain;
%         end
%         Dsar(i)         = 0;                     % Final SAR output
% 
%         for j           = bit:-1:1
%             if gain*Vres(i)+4*Vint(i)  > 0
%                 D(j)    = 1;                    % D1 is SAR ADC digital output array
%             else
%                 D(j)    = -1;
%             end
%             Vres(i)     = Vres(i) - D(j) * C(j)/Ctot;        % SAR residue
%             Dsar(i)     = Dsar(i) + D(j) * 2^(j-bit-1); 
%         end
% 
%         Dlsb(i)         = Dsar(i) - D(bit)/2;
%         Vlsb(i)         = sum(D.*C)/Ctot - D(bit)*C(bit)/Ctot; 
%         Dout(i)         = Dsar(i) - Dlsb(i-1) + Dp(i)/2; % Dout = Dsar - Dlsb*z^(-1) + Dp/2
%         Vs(i)           = Vin(i)-Dout(i);
%     end

%end of ADC, Dout now is output of ADC
%%%%%%  scalling    
%  for i = 1:length(Dout)
%     Dout(i)=log2(abs(Dout(i)));
%  end
%  Dout=Dout/8;

%% plot
% selected_channel_number = 1;
% trial_num = 1;
% 
% figure();
% % plot(time_in_ms, data_in(trial_num,:,selected_channel_number)/1000);
% % hold on;
% plot(time_in_ms, Dout(1,1:7501));
% hold on;
% plot(time_in_ms, Vin(1,1:7501));
% hold off;
% 
% legend('Real signal with artifacts', 'ADC output signal');
% xlabel('Time (ms)');
% ylabel('Amplitude (mV)');
% grid on;


%%  template detection 
% hyperparameter if scale to [-1,1]
% pthreshold    =0.05;
% nthreshold    =-0.05;
% flag_threshold = 0.2;
% hyperparameter if no scaling
pthreshold    = 5;
nthreshold    = -5;
flag_threshold = 200;

Dtemp_1 = zeros(total_trials, num_timesteps, total_channels); % method 1: 下降
% Dtemp_2 = zeros(total_trials, num_timesteps, total_channels); % method 2：上升

Dtemp_1(:, 1:1000,:) = data_in(:, 1:1000,:); % no artifact period
Dtemp_1(:, 5000:7501,:) = data_in(:, 5000:7501,:); % no artifact period

for trial_num = 1:total_trials
    Ain_t=squeeze(data_in(trial_num,:,:));
      
    for ch = 1:total_channels
        Ain = Ain_t(:,ch);
        Ain = Ain';
        Dout=Ain;
        flag_flat                = zeros(1,num_timesteps);
        flag_1                   = zeros(1,num_timesteps);
        
        for i = 2:length(Dout)
        % if -threshold<Dout(i)-Dout(i-1)<threshold
        %    Dinte(i)=Dout(i);
            if Dout(i)-Dout(i-1) < nthreshold
              flag_1(i)=1;
              if flag_flat(i-1)==1 
                  if Dout(i-1)< flag_threshold && Dout(i-1)>-flag_threshold
                  Dtemp_1(trial_num,i,ch)=Ain(i-1);% Vin instead of Ain for ADC
                  flag_flat(i)=0;
                  else 
                      Dtemp_1(trial_num,i,ch)=0;
                  end
              end
            end
            if Dout(i)-Dout(i-1) < pthreshold && Dout(i)-Dout(i-1) > nthreshold
              flag_flat(i)=1;
                if flag_1(i-1)==1 
                    if Dout(i)< flag_threshold && Dout(i)>-flag_threshold
                   % Dtemp_2(i,ch)=Ain(i); % Vin instead of Ain for ADC
                   flag_1(i)=0;
                    % else
                    %     Dtemp_2(i,ch)=0;
                    end
                end
            end
        end
        % Find indices of zero values
        zeroIndices = find(Dtemp_1(trial_num,:,ch) == 0);
        % zeroIndices_2 = find(Dtemp_2(:,ch) == 0);
        % Find indices of non-zero values
        nonZeroIndices = find(Dtemp_1(trial_num,:,ch) ~= 0);
        % nonZeroIndices_2 = find(Dtemp_2(:,ch) ~= 0);

        % Perform linear interpolation at zero indices        
        if ~isempty(nonZeroIndices)
            Dtemp_1(trial_num,zeroIndices,ch) = interp1(nonZeroIndices, Dtemp_1(trial_num,nonZeroIndices,ch), zeroIndices, 'linear', 'extrap');
        else
            Dtemp_1(trial_num,zeroIndices,ch) = 0; % Or another default value
        end

        % Dtemp_2(zeroIndices_2,ch) = interp1(nonZeroIndices_2, Dtemp_2(nonZeroIndices_2,ch), zeroIndices_2, 'linear');
    end
end


% plot
selected_trial_number = 1;
selected_channel_number = 1;
% Create main figure
fig = figure();
mainAxes = axes('Parent', fig);
% Plot data in main axes
hold(mainAxes, 'on');
plot(mainAxes, data_in(selected_trial_number,:,selected_channel_number));
plot(mainAxes, Dtemp_1(selected_trial_number,:,selected_channel_number));
hold(mainAxes, 'off');
legend(mainAxes, 'real signal with artifacts', 'reconstructed signal 1');
grid(mainAxes, 'on');



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