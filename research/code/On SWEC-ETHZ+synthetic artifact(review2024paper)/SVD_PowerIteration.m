clear;
clear variables; 
close all;
clc;

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

[total_trial_num, num_timesteps, total_channels] = size(data_in);


% Hyperparameters
N = 2; % Number of largest singular values to remove
stimulation_timerange = 1:num_timesteps;
svd_window_size = 1000; % pulse width is 90 timesteps = 3 ms
num_windows = ceil(length(stimulation_timerange) / svd_window_size);
svd_channel_size = 88;
num_channels_groups = ceil(total_channels / svd_channel_size);
iter_num = 20; % number of power iteration steps
reconstructed_signal = data_in;


%% Power iteration method
for trial_num = 1:total_trial_num
    Ain_t =  squeeze(data_in(trial_num,:,:));
    
    for window_idx = 1:num_windows
        % Determine the start and end of the current window
        start_idx = (window_idx - 1) * svd_window_size + 1;
        end_idx = min(window_idx * svd_window_size, length(stimulation_timerange));
        
        % Extract the windowed data
        window_range = start_idx:end_idx;
        A_window = Ain_t(window_range, :);
        
        for ch_idx = 1:num_channels_groups
            % way 1: group channels 1-8, 9-16, ... 
            start_idx = (ch_idx - 1) * svd_channel_size + 1;
            end_idx = min(ch_idx * svd_channel_size, total_channels);
            channel_range = start_idx:end_idx;

            % way 2: [1,12,23,34,45,56,67,78], ...
%             start_offset = ch_idx -1;
%             channel_pattern = [1, 12, 23, 34, 45, 56, 67, 78]; % Example pattern
%             channel_range = start_offset + channel_pattern;
%             channel_range = channel_range(channel_range <= total_channels);

            A_window_channel = A_window(:,channel_range);
            % Perform Power Iteration SVD
            reconstructed_window_channel = power_iteration(A_window_channel, N, iter_num); 

            % Update the corresponding part of the reconstructed signal
            reconstructed_signal(trial_num, stimulation_timerange(window_range), channel_range) = reconstructed_window_channel;

        end
        
    end
end
% if seizure_flag
%     save('/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/ethz_data/interp/OldData_SVD_AcrossChannels_N1_seizure_amp73.mat', 'reconstructed_signal');
% else
%     save('/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/ethz_data/interp/SVD_AcrossChannels_N1_nonseizure_amp73.mat', 'reconstructed_signal');
% end

% Calculate and display some comparison metrics
[mse] = SynGT_performance_metrics_allTrials(synthetic_GT, reconstructed_signal);
%% Plot results
% data = load('/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/ethz_data/interp/OldData_ASAR_seizure_amp73.mat');
% Dout_clean = data.Dout_clean;
% 
% data = load('/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/ethz_data/interp/OldData_interpolation_seizure_amp73.mat');
% Dtemp_1 = data.Dtemp_1;

selected_channel_number = 1;
selected_trial_number = 1;
figure();
% plot(time_in_ms, Dout_clean(selected_trial_number, :,selected_channel_number)/1e3,  'LineWidth', 2, 'Color', [0.8, 0.8, 0.8]);
% hold on;
% plot(time_in_ms, Dtemp_1(selected_trial_number, :,selected_channel_number)/1e3-3.2, 'LineWidth', 3,'Color', [1, 0, 0]); % 'Color', [0.5, 0, 0.5]);
% hold on;
plot(time_in_ms, reconstructed_signal(selected_trial_number, :,selected_channel_number)/1e3, 'LineWidth', 2, 'Color', [0, 0.5, 0]);
hold on;
plot(time_in_ms, synthetic_GT(selected_trial_number, :,selected_channel_number)/1e3, LineWidth=2.5, Color='blue');
hold off;
grid on;
ylim([-0.5, 0.5])
xlim([0 4000])
% legend( 'Apply ASAR', 'Apply Interpolation', 'Apply SVD', 'GT seizure signal','FontSize', 16);
legend('Apply SVD across channels', 'GT seizure signal','FontSize', 16);
xlabel('Time (ms)', 'FontSize', 16);
ylabel('Voltage (mV)', 'FontSize', 16);
set(gca, 'FontSize', 16);  % Set font size for tick labels

%% Function to perform Power Iteration for SVD
function A_copy = power_iteration(A, num_vecs, num_iters)
    [m, n] = size(A);
    U = zeros(m, num_vecs);
    S = zeros(num_vecs, num_vecs);
    V = zeros(n, num_vecs);
    
    A_copy = A;
    
    for i = 1:num_vecs
        v = randn(n, 1); % Random initialization v
        v = v / norm(v);

        % calculate B=A^T A
        B = A_copy' * A_copy;
        
        % calculate B^k * v / ||B^k * v||
        v = B^num_iters * v;
        v = v / norm(v);
        
        sigma = norm(A_copy * v);
        % make sure sigma is positive value
        if sigma < 0
            sigma = -sigma;
            v = -v;
        end
        
        U(:, i) = A_copy * v / sigma;
        S(i, i) = sigma;
        V(:, i) = v;
        
        % Deflation: Remove the contribution of this singular component
        A_copy = A_copy - S(i, i) * (U(:, i) * V(:, i)');
    end
end