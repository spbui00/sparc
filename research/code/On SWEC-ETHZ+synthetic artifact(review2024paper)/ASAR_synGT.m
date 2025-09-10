clear;
clc;
close all;


%% SWEC-ETHZ iEEG dataset
seizure_flag = true;
data_folder = 'high_amp_73/';
% data_folder = 'amp_57/';
if seizure_flag
% % %     load(fullfile(data_folder, 'mixed_rate512Hz.mat'));
% % %     load(fullfile(data_folder, 'clean_rate512Hz.mat'));
%     load(fullfile(data_folder, 'mixed_seizure1_rate2kHz.mat'));
%     load(fullfile(data_folder, 'clean_seizure1_rate2kHz.mat'));
    load(fullfile(data_folder, 'swec-ethz-ieeg-seizure-data-rate512Hz.mat'));
    data_in = mixed_seizure;
    synthetic_GT = signal_seizure;
else
%     load(fullfile(data_folder, 'swec-ethz-ieeg-nonseizure-data-rate512Hz.mat'));
%     data_in = mixed_nonseizure;
%     synthetic_GT = signal_nonseizure;
    load(fullfile(data_folder, 'mixed_nonseizure1_rate2kHz.mat'));
    load(fullfile(data_folder, 'clean_nonseizure1_rate2kHz.mat'));
    
    data_in = mixed_nonseizure;
    synthetic_GT = signal_nonseizure;
end

data_in = permute(data_in, [1,3,2]); %convert to [trials, timesteps, channels]
synthetic_GT = permute(synthetic_GT, [1,3,2]); %convert to [trials, timesteps, channels]

% Create time vector in milliseconds
sampling_rate = 512; 
time_in_ms = (0:size(data_in, 2)-1) / sampling_rate * 1000; % Convert to milliseconds



%% ASAR
N=2052; % first N timesteps used to estimate mean and std of clean signal

% low amp filter parameters
% filterLength =201;  % Length of the adaptive filter
% mu1 =0.1;          % Adaptation step size
% a= 0.5; % threshold for keeping the original signal

% % high amp filter parameters for non-seizure
% filterLength =16;  % Length of the adaptive filter
% mu1 = 0.5;          % Adaptation step size
% a= 10; % threshold for keeping the original signal

% high amp filter parameters for non-seizure
filterLength =16;  % Length of the adaptive filter
mu1 = 0.9;          % Adaptation step size
a= 10; % threshold for keeping the original signal


[trial_num, ncycle, channel_num] = size(data_in);
Dout_clean = zeros(trial_num, ncycle, channel_num);

for trial = 1:trial_num
    Ain_t = squeeze(data_in(trial,:,:));
    for ch = 1:channel_num
        Ain = Ain_t(:,ch);
        Ain = Ain';
        Dout=Ain;
        if ch < channel_num
            Dout_nearby = Ain_t(:,ch+1);
        else
            Dout_nearby = Ain_t(:,ch-1);
        end
    
        synthetic_GT_ch = synthetic_GT(trial,:,ch);
        synthetic_GT_ch = synthetic_GT_ch';
    
        % Initialize filter coefficients
        w1 = zeros(filterLength, 1);
        S=0;
        T=0;
        u = zeros(1,filterLength);
        
        if N > 0 
            for n =1:N  
                 S=S+synthetic_GT_ch(n);
                 T=T+(synthetic_GT_ch(n)^2);
            end
            avg=S/N;
            std=sqrt((1/(N-1))*(T-(N*(avg^2))));
        end

        % Apply adaptive filtering for artifact cancellation
        Dout2=Ain;
        for n =1:ncycle
                
                u(1,2:end) = u(1,1:end-1);  % Shifting of frame window
                
                if N == 0
                    u(1,1)=Dout_nearby(n);
                elseif (abs(Dout_nearby(n)-avg)>=a*std)
                    u(1,1)=Dout_nearby(n);
                else
                    u(1,1)=0;
                end 
                    
                    adaptive_filter_out1(n) =  u*w1;
                    error1(n) =  Dout2(n) - adaptive_filter_out1(n);
                    w1 = w1 + (mu1 * u' * error1(n)/(u*u'+0.0001));
                    %w1 = w1 + (mu1 * u' * error1(n));
                    Dout_clean(trial, n, ch)=Dout2(n)- u*w1;
        end
    
    end
end
if seizure_flag
    save('/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/ethz_data/interp/OldData_ASAR_seizure_amp73.mat', 'Dout_clean');
else
    save('/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/ethz_data/interp/ASAR_nonseizure_amp73.mat', 'Dout_clean');
end
% Calculate and display some comparison metrics
[mse, psd] = SynGT_performance_metrics_allTrials(synthetic_GT, Dout_clean);

%% plot
selected_clip_number = 1;
selected_channel_number = 1;
figure();
plot(time_in_ms, squeeze(synthetic_GT(selected_clip_number,:,selected_channel_number))/1e3);
hold on;
plot(time_in_ms, squeeze(Dout_clean(selected_clip_number,:,selected_channel_number))/1e3);
hold off;
legend('GT clean signal','After ASAR denoised signal')
grid on;
xlabel('Time (ms)');
ylabel('Voltage (mV)');





%%  DBS data set
% Nin = par.ain*sin(2*pi*par.fin*t)';
% load('+data/50ad9_paramSweep4.mat') 
% fsData = fs_data;
% tEpoch = t_epoch;
% %xlims = [-200 600];
% chanIntList = [5,6,7,9,10];
% trainDuration = [0 500];
% minDuration = 0.250; % minimum duration of artifact in ms
% Ain = 4*dataInt(:,1,1); % needed to be multiplied by 4 from raw recordin

%% Generate random Fourier feature weights and biases
% % Define parameters
% original_dimension = 100; % Dimensionality of the original data
% num_random_features = 5000; % Number of random Fourier features
%omega = randn(original_dimension, num_random_features);
% omega=Ain(1:original_dimension);
% b = 2 * pi * rand(1, num_random_features);
% 
% % Generate random data vector
% data_vector = randn(original_dimension, 1);
% 
% % Project data into random Fourier feature space
% projected_data = sqrt(2/num_random_features) * cos(omega' * data_vector + b');

% Display the results
% legend('Input','clean');
% offset = 7;
% figure(2)
% plot( projected_data)
% hold on
% plot( Ain(1:num_random_features))
% hold off

% Apply the FIR filter to the input signal
%Lp_Ain = filter(filter_coeffs, 1, Ain);


