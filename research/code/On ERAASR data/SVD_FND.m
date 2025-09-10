clear;
clear variables; 
close all;
clc;

%% Stanford dataset
dat=load('exampleDataTensor.mat');
data_in=dat.data_trials_by_time_by_channels;

total_trials = size(data_in,1);
total_channels = size(data_in,3);
num_timesteps = size(data_in,2);

% Create time vector in milliseconds
sampling_rate = 30000; % 30 kHz
time_in_ms = (0:size(data_in, 2)-1) / sampling_rate * 1000; % Convert to milliseconds


% downsample plot
selected_channel_number = 1;
trial_num = 1;
x = data_in(trial_num,:,selected_channel_number)/1000;
desired_fs = 1000; 
downsample_factor = sampling_rate / desired_fs;
x_downsampled = decimate(x, downsample_factor);

t_downsampled = time_in_ms(1:downsample_factor:end);

% Plot the original and downsampled signals
figure;
subplot(2,1,1);
plot(time_in_ms, x);
title('Original Signal (30 kHz)');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,1,2);
plot(t_downsampled, x_downsampled);
title('Downsampled Signal (1 kHz)');
xlabel('Time (s)');
ylabel('Amplitude');

%% SVD
% % hyperparameters
% N = 5; % Replace with the desired number of largest singular values to zero out
% stimulation_timerange = 1501:5000;
% reconstructed_signal = data_in;
% 
% for trial_num = 1:total_trials
%     Ain_t=squeeze(data_in(trial_num,stimulation_timerange,:));
%     % Perform SVD 
%     [U, S, V] = svd(Ain_t);
%     
%     % Sort singular values in descending order and get their indices
%     [sorted_values, sort_indices] = sort(diag(S), 'descend');
%     
%     % Determine which indices correspond to the N largest values
%     largest_N_indices = sort_indices(1:N);
%     
%     % Create a copy of S with the N largest singular values zeroed out
%     S_filtered = S;
%     S_filtered(largest_N_indices, largest_N_indices) = 0;
%     
%     % Reconstruct the signal with removed components
%     reconstructed_signal(trial_num,stimulation_timerange,:) = U * S_filtered * V';
% end

%% SVD component check
% % hyperparameters
% N = 5; % Replace with the desired number of largest singular values to zero out
% stimulation_timerange = 1501:5000;
% trial_num = 20;
% 
% reconstructed_signal = data_in;
% Ain_t=squeeze(data_in(trial_num,stimulation_timerange,:));
% % Perform SVD 
% [U, S, V] = svd(Ain_t);
% 
% % Sort singular values in descending order and get their indices
% [sorted_values, sort_indices] = sort(diag(S), 'descend');
% 
% % Determine which indices correspond to the N largest values
% largest_N_indices = sort_indices(1:N);
% 
% % Create a copy of S with the N largest singular values zeroed out
% S_filtered = S;
% S_filtered(largest_N_indices, largest_N_indices) = 0;
% 
% % Reconstruct the signal with removed components
% reconstructed_signal(trial_num,stimulation_timerange,:) = U * S_filtered * V';


%% SVD on small time windows
% Hyperparameters
N = 10; % Number of largest singular values to zero out
stimulation_timerange = 1501:5000;
svd_window_size = 20; % inter pulse intervval is 90 timesteps = 3 ms
num_windows = ceil(length(stimulation_timerange) / svd_window_size);
reconstructed_signal = data_in;

for trial_num = 1:total_trials
    Ain_t = squeeze(data_in(trial_num, stimulation_timerange, :));
       
    for window_idx = 1:num_windows
        % Determine the start and end of the current window
        start_idx = (window_idx - 1) * svd_window_size + 1;
        end_idx = min(window_idx * svd_window_size, length(stimulation_timerange));
        
        % Extract the windowed data
        window_range = start_idx:end_idx;
        A_window = Ain_t(window_range, :);
        
        % Perform SVD on the windowed data
        [U, S, V] = svd(A_window, 'econ'); % Use 'econ' for efficiency on non-square matrices
        
        % Zero out the N largest singular values
        S_filtered = S;
        S_filtered(1:N, 1:N) = 0;
        
        % Reconstruct the signal for the current window
        reconstructed_window = U * S_filtered * V';
        
        % Update the corresponding part of the reconstructed signal
        reconstructed_signal(trial_num, stimulation_timerange(window_range), :) = reconstructed_window;
    end
end

%% SVD on small time windows, adaptive N 
% % Hyperparameters
% 
% lower_bound = 0.6; % Define stopping range
% upper_bound = 1/lower_bound; % Define stopping range
% 
% stimulation_timerange = 1501:5000;
% svd_window_size = 90; % inter pulse intervval is 90 timesteps = 3 ms
% num_windows = ceil(length(stimulation_timerange) / svd_window_size);
% 
% reconstructed_signal = data_in;
% 
% k_removed = zeros(total_trials,num_windows);
% 
% for trial_num = 1:total_trials
%     Ain_t = squeeze(data_in(trial_num, stimulation_timerange, :));
%        
%     for window_idx = 1:num_windows
%         % Determine the start and end of the current window
%         start_idx = (window_idx - 1) * svd_window_size + 1;
%         end_idx = min(window_idx * svd_window_size, length(stimulation_timerange));
%         
%         % Extract the windowed data
%         window_range = start_idx:end_idx;
%         A_window = Ain_t(window_range, :);
%         
%         % Perform SVD on the windowed data
%         [U, S, V] = svd(A_window, 'econ'); % Use 'econ' for efficiency on non-square matrices
%         
%         % Extract singular values
%         singular_values = diag(S);
%         num_singular_values = length(singular_values);
%         % Initialize k
%         k = 1;    
%         % Loop through singular values to find the stopping point
%         while k < num_singular_values - 2
%             % Compute relative drop ratio
%             ratio = (singular_values(k) - singular_values(k+1)) / (singular_values(k+1) - singular_values(k+2));     
%             % Stop when ratio falls within the range
%             if lower_bound <= ratio && ratio <= upper_bound
%                 break;
%             end
%             
%             k = k + 1;
%         end
%         k_removed(trial_num, window_idx) = k;
%         % Zero out the top k singular values
%         S_filtered = S;
%         S_filtered(1:k, 1:k) = 0;
% 
%         
%         % Reconstruct the signal for the current window
%         reconstructed_window = U * S_filtered * V';
%         
%         % Update the corresponding part of the reconstructed signal
%         reconstructed_signal(trial_num, stimulation_timerange(window_range), :) = reconstructed_window;
%     end
% end

%% plot
selected_channel_number = 1;
trial_num = 1;

figure();
plot(time_in_ms, data_in(trial_num,:,selected_channel_number)/1000);
hold on;
plot(time_in_ms, reconstructed_signal(trial_num,:,selected_channel_number)/1000);
hold off;

legend('Real signal with artifacts', 'Reconstructed signal');
xlabel('Time (ms)');
ylabel('Amplitude (mV)');
grid on;



