clear;
clc;
close all;

% load original data
dat=load('exampleDataTensor.mat');
data_in=dat.data_trials_by_time_by_channels;
Ain_t=squeeze(data_in(1,:,:));


% synthetic data load
dat=load('SyntheticNeuralSig.mat');
synthetic_GT = dat.synthetic_neural_signal;
synthetic_GT = squeeze(synthetic_GT(1,:,:));

% Define filter parameters
filterLength =201;  % Length of the adaptive filter
mu1 =0.1;          % Adaptation step size
a= 5; % threshold for keeping the original signal
ncycle = size(Ain_t,1);
Dout_clean = zeros(ncycle, 5);

for ch = 1:5
    Ain = Ain_t(:,ch);
    Ain = Ain';
    Dout=Ain;
    % Initialize filter coefficients
    w1 = zeros(filterLength, 1);
    S=0;
    T=0;
    u = zeros(1,filterLength);
    N=500;
    for n =1:N  
         S=S+Dout(n);
         T=T+(Dout(n)^2);
    end
    avg=S/N;
    std=sqrt((1/(N-1))*(T-(N*(avg^2))));
    % Apply adaptive filtering for artifact cancellation
    Dout2=Ain;
    
    for n =1:ncycle
            
            u(1,2:end) = u(1,1:end-1);  % Shifting of frame window
    
            if(abs(Dout(n)-avg)>=a*std)
                u(1,1)=Dout(n);
            else
                u(1,1)=0;
            end 
                
                adaptive_filter_out1(n) =  u*w1;
                error1(n) =  Dout2(n) - adaptive_filter_out1(n);
                w1 = w1 + (mu1 * u' * error1(n)/(u*u'+0.0001));
                %w1 = w1 + (mu1 * u' * error1(n));
                Dout_clean(n,ch)=Dout2(n)- u*w1;
    end

end

% Calculate and display some comparison metrics
[mse, psd_mse_values] = SynGT_performance_metrics(synthetic_GT, Dout_clean);

% plot
selected_channel_number = 1;
figure();
plot(squeeze(synthetic_GT(:,selected_channel_number)));
hold on;
plot(Ain_t(:,selected_channel_number));
hold on;
plot(Dout_clean(:,selected_channel_number));
hold off;
legend('synthetic GT without artifacts','real signal with artifacts','reconstructed signal')
grid on;





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


