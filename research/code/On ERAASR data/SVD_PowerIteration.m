clear;
clear variables; 
close all;
clc;

%% Stanford dataset
dat = load('exampleDataTensor.mat');
data_in = dat.data_trials_by_time_by_channels;

total_trials = size(data_in,1);
total_channels = size(data_in,3);
num_timesteps = size(data_in,2);

% Create time vector in milliseconds
sampling_rate = 30000; % 30 kHz
time_in_ms = (0:size(data_in, 2)-1) / sampling_rate * 1000; % Convert to milliseconds

% Hyperparameters
N = 10; % Number of largest singular values to remove
stimulation_timerange = 1501:5000;
svd_window_size = 20; % pulse width is 90 timesteps = 3 ms
num_windows = ceil(length(stimulation_timerange) / svd_window_size);
reconstructed_signal = data_in;


%% Iterating over trials
for trial_num = 1:total_trials
    Ain_t = squeeze(data_in(trial_num, stimulation_timerange, :));
    
    for window_idx = 1:num_windows
        % Determine the start and end of the current window
        start_idx = (window_idx - 1) * svd_window_size + 1;
        end_idx = min(window_idx * svd_window_size, length(stimulation_timerange));
        
        % Extract the windowed data
        window_range = start_idx:end_idx;
        A_window = Ain_t(window_range, :);
        
        % Perform Power Iteration SVD
        reconstructed_window = power_iteration(A_window, N, 20); % 20 iterations
        
        % Update the corresponding part of the reconstructed signal
        reconstructed_signal(trial_num, stimulation_timerange(window_range), :) = reconstructed_window;
    end
end

%% Plot results
selected_channel_number = 1;
trial_num = 1;

figure();
plot(time_in_ms, data_in(trial_num,:,selected_channel_number));
hold on;
plot(time_in_ms, reconstructed_signal(trial_num,:,selected_channel_number));
hold off;

legend('Real signal with artifacts', 'Reconstructed signal');
xlabel('Time (ms)');

% Function to perform Power Iteration for SVD
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