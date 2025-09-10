clear;
clear variables; 
close all;
clc;

%% SWEC-ETHZ iEEG dataset
seizure_flag = true;
sampling_rate = 512; 
template_length_timesteps = 40; % amp57 = 16, amp73 = 128
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
    load(fullfile(data_folder, 'swec-ethz-ieeg-nonseizure-data-rate512Hz.mat'));
    data_in = mixed_nonseizure;
    synthetic_GT = signal_nonseizure;
end
% permute dim
data_in = permute(data_in, [1,3,2]); %convert to [trials, timesteps, channels]
synthetic_GT = permute(synthetic_GT, [1,3,2]); %convert to [trials, timesteps, channels]



% Ain_t = squeeze(data_in(select_clip_num,:,:));
% synthetic_GT = squeeze(synthetic_GT(select_clip_num,:,:));
Ain_ch = data_in;



% reshape along pulse
total_timesteps = size(Ain_ch, 2);
n_cycle = floor(total_timesteps / template_length_timesteps);
residual = total_timesteps - n_cycle * template_length_timesteps;
% cut residual for both
Ain_ch = Ain_ch(:,1:total_timesteps-residual,:);
synthetic_GT = synthetic_GT(:,1:total_timesteps-residual,:);

% Create time vector in milliseconds
time_in_ms = (0:size(Ain_ch, 2)-1) / sampling_rate * 1000; % Convert to milliseconds

Ain_4d = reshape(Ain_ch, size(Ain_ch,1), template_length_timesteps, n_cycle, size(Ain_ch,3));
[num_trials, template_length_timesteps, n_cycle, num_channels] = size(Ain_4d);
Ain_pulse = reshape(permute(Ain_4d, [2, 1, 3, 4]), template_length_timesteps, []);


% Plotting pulses to check
num_pulses_to_plot = min(5, size(Ain_pulse, 2));  % Plot up to 5 pulses
selected_pulses = randperm(size(Ain_4d, 2), num_pulses_to_plot);
% Create the plot
figure('Position', [100, 100, 1200, 600]);
hold on;
% Color palette for different pulses
colors = [
    0.0, 0.4470, 0.7410;  % Blue
    0.8500, 0.3250, 0.0980;  % Orange
    0.9290, 0.6940, 0.1250;  % Yellow
    0.4940, 0.1840, 0.5560;  % Purple
    0.4660, 0.6740, 0.1880   % Green
];
% Plot selected pulses
for i = 1:length(selected_pulses)
    pulse_index = selected_pulses(i);
    pulse_data = squeeze(Ain_pulse(:, pulse_index));
    
    plot(1:length(pulse_data), pulse_data, ...
        'LineWidth', 2, ...
        'Color', colors(mod(i-1, size(colors,1))+1, :), ...
        'DisplayName', sprintf('Pulse %d', pulse_index));
    hold on;
end
title('Randomly Selected Pulses');
xlabel('Time Points within Pulse');
ylabel('Amplitude');
legend('show', 'Location', 'best');
grid on;
hold off;
% 
% 
% % reshape back
% Ain_restored = reshape(Ain_pulse, size(permute(Ain_4d, [2, 1, 3, 4]))); % [template_length_timesteps, num_trials, n_cycle, num_channels]
% Ain_restored_same4d = permute(Ain_restored,[2, 1, 3, 4]); % [num_trials, template_length_timesteps, n_cycle, num_channels]
% Ain_restored = permute(Ain_restored_same4d,[1, 4, 2, 3]);% [num_trials, num_channels, template_length_timesteps, n_cycle]
% Ain_restored = reshape(Ain_restored, num_trials, num_channels, []); % [num_trials, num_channels, totaltimesteps]
% Ain_restored = permute(Ain_restored, [1, 3, 2]); % [num_trials, totaltimesteps, num_channels]
% Verify
% disp('Original 3D matrix size:');
% disp(size(Ain_ch));
% disp('Restored matrix size:');
% disp(size(Ain_restored));
% % Check if the matrices are equivalent
% max_diff = max(abs(Ain_ch(:) - Ain_restored(:)));
% disp('Maximum difference between original and restored:');
% disp(max_diff);
%% SVD
% Parameters
K = 10;                 % Number of chunks
N = 3;                  % Number of top singular values to remove
[timesteps, total_trials] = size(Ain_pulse);
chunk_size = floor(total_trials / K);
reconstructed_signal = [];
save_component = false;

for k = 1:K
    % Determine the start and end columns of the chunk
    start_idx = (k - 1) * chunk_size + 1;
    if k == K
        end_idx = total_trials; % Last chunk takes the remainder
    else
        end_idx = k * chunk_size;
    end

    % Extract chunk
    chunk = Ain_pulse(:, start_idx:end_idx);

    % Perform SVD
    [U, S, V] = svd(chunk, 'econ');
    singular_values = diag(S);
    
    % Sort and zero out top N singular values
    [~, sort_indices] = sort(singular_values, 'descend');
    S_filtered = S;
    for i = 1:min(N, length(singular_values))
        idx = sort_indices(i);
        S_filtered(idx, idx) = 0;
    end

    % Reconstruct chunk
    chunk_reconstructed = U * S_filtered * V';

    % Concatenate the result
    reconstructed_signal = [reconstructed_signal, chunk_reconstructed];
end



% Optionally, save the deleted components
if save_component
    component_matrices = cell(1, N);
    for i = 1:N
        % Create a temporary S matrix with only one singular value
        S_temp = zeros(size(S));
        S_temp(sort_indices(i), sort_indices(i)) = sorted_values(i);        
        % Reconstruct the component matrix
        component_matrices{i} = U * S_temp * V';
        
        % reshape back
        component_matrices{i} = reshape(component_matrices{i}, size(permute(Ain_4d, [2, 1, 3, 4]))); % [template_length_timesteps, num_trials, n_cycle, num_channels]
        component_matrices{i} = permute(component_matrices{i},[2, 1, 3, 4]); % [num_trials, template_length_timesteps, n_cycle, num_channels]
        component_matrices{i} = permute(component_matrices{i},[1, 4, 2, 3]);% [num_trials, num_channels, template_length_timesteps, n_cycle]
        component_matrices{i} = reshape(component_matrices{i}, num_trials, num_channels, []); % [num_trials, num_channels, totaltimesteps]
        component_matrices{i} = permute(component_matrices{i}, [1, 3, 2]); % [num_trials, totaltimesteps, num_channels]
    end
end

% Reshape back 
Ain_restored = reshape(reconstructed_signal, size(permute(Ain_4d, [2, 1, 3, 4]))); % [template_length_timesteps, num_trials, n_cycle, num_channels]
Ain_restored_same4d = permute(Ain_restored,[2, 1, 3, 4]); % [num_trials, template_length_timesteps, n_cycle, num_channels]
Ain_restored = permute(Ain_restored_same4d,[1, 4, 2, 3]);% [num_trials, num_channels, template_length_timesteps, n_cycle]
Ain_restored = reshape(Ain_restored, num_trials, num_channels, []); % [num_trials, num_channels, totaltimesteps]
reconstructed_signal = permute(Ain_restored, [1, 3, 2]); % [num_trials, totaltimesteps, num_channels]

%% Calculate and display some comparison metrics
% for the select clip
% select_clip_num = 1;
% Ain_ch = squeeze(Ain_ch(select_clip_num,:,:));
% reconstructed_signal = squeeze(reconstructed_signal(select_clip_num,:,:));
% synthetic_GT = squeeze(synthetic_GT(select_clip_num,:,:));
% if save_component
%     for i=1:N
%         component_matrices{i} = squeeze(component_matrices{i}(select_clip_num,:,:));
%     end
% end
save('/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/ethz_data/interp/OldData_SVD_AcrossPulses_N1.mat', 'reconstructed_signal');
[mse, psd] = SynGT_performance_metrics_allTrials(synthetic_GT, reconstructed_signal);

% Optional: Calculate MSE and PSD MSE for the benchmark window
% mse_benchmark = mean((synthetic_GT(benchmark_window,:) - Ain_t(benchmark_window,1:5)).^2); 
% psd_mse_benchmark = zeros(5, 1);
% for i = 1:5
%     [psd_synthetic, f] = pwelch(synthetic_GT(benchmark_window,i), hanning(nfft), noverlap, nfft, fs);
%     [psd_original, ~] = pwelch(Ain_t(benchmark_window,i), hanning(nfft), noverlap, nfft, fs);
%     psd_mse_benchmark(i) = mean((psd_synthetic - psd_original).^2);
% end
% 
% fprintf("\nMSE for benchmark window (1:1500):\n");
% fprintf('%f\n', mse_benchmark);
% fprintf("\nPSD MSE values for benchmark window (1:1500):\n");
% fprintf('%f\n', psd_mse_benchmark);



selected_channel_number = 1;
selected_trial_number = 1;
figure();
plot(time_in_ms, reconstructed_signal(selected_trial_number, :,selected_channel_number)/1e3, 'LineWidth', 2, 'Color', [0, 0.5, 0]);
hold on;
plot(time_in_ms, synthetic_GT(selected_trial_number, :,selected_channel_number)/1e3, LineWidth=2.5, Color='blue');
hold off;
grid on;
ylim([-0.5, 0.5])
xlim([0 4000])
legend('Apply SVD across pulses', 'GT seizure signal','FontSize', 16);
xlabel('Time (ms)', 'FontSize', 16);
ylabel('Voltage (mV)', 'FontSize', 16);
set(gca, 'FontSize', 16);  % Set font size for tick labels



%% Generate artifact
% % Synthetic artifact Parameters
% stim_rate = 1000;                % Stimulation rate (Hz)
% sampling_rate_signal = 20000;    % Sampling rate (Hz)
% stim_period = 1 / stim_rate;     % Stimulation period (s)
% dt = 1 / sampling_rate_signal;   % Time step (s)
% Total_time = 4;                  % Total duration (s)
% f_pulse = 2500;                  % Pulse frequency (Hz)

% low freq component computation
% fs_high = sampling_rate_signal;  % High sampling rate (20 kHz)
% fs_low = sampling_rate;     % Target sampling rate (512 Hz)
% time_length = 4;  % Signal duration in seconds
% t_high = 0:1:time_length*fs_high;  % High-rate time vector
% low_freq = low_freq_scaler_mV * sin(2 * pi * 0.01 * t_high);
% 
% % Downsample the signal 
% factor = floor(fs_high / fs_low);
% low_freq_downsampled = low_freq(1:factor:size(low_freq,2));
% low_freq_downsampled = low_freq_downsampled(1:n_cycle*template_length_timesteps);
% 

% %parameters
% k2 = 0.102;
% stim_current_strength = 73;
% low_freq_scaler_mV = 5;
% % artifact template component
% amplitude_scaler = 10^(k2*stim_current_strength - 1.92);
% time_artifact = 0:dt:Total_time-dt;
% clip_artifact = zeros(size(time_artifact));
% 
% for start_time = 0:stim_period:time_artifact(end)
%     % Find indices for sine wave component
%     indices_sine = (time_artifact >= start_time) & (time_artifact < start_time + 1 / f_pulse);
%     
%     % Generate random delay for exponential component
%     rand_delay = rand * (7/8 * (1/f_pulse) - 3/8 * (1/f_pulse)) + 3/8 * (1/f_pulse);
%     indices_exp = (time_artifact >= start_time + rand_delay);
%     
%     % Apply sine wave artifact
%     clip_artifact(indices_sine) = -sin(2 * pi * f_pulse * (time_artifact(indices_sine)-start_time));
%     
%     % Apply exponential decay artifact
%     clip_artifact(indices_exp) = clip_artifact(indices_exp) - exp(-3000 * (time_artifact(indices_exp)-start_time)) - exp(-5000 * (time_artifact(indices_exp)-start_time));
% 
% %     figure;plot(clip_artifact(1:160));
% end
% clip_artifact = clip_artifact * amplitude_scaler;
% % Downsample the signal 
% clip_artifact_downsampled = clip_artifact(1:factor:size(clip_artifact,2));
% clip_artifact_downsampled = clip_artifact_downsampled(1:n_cycle*template_length_timesteps);

% Plot the removed signal
% figure;
% for i=1:N
%     subplot(N+2,1,i);
%     plot(time_in_ms, component_matrices{i}(:,1)/1e3);
%     title(sprintf('Removed component %d ', i));
%     xlabel('Time (ms)');
%     ylabel('Amplitude (mV)');
% end

% subplot(N+2,1,N+1);
% plot(time_in_ms, low_freq_downsampled);
% title('low freq component');
% xlabel('Time (ms)');
% ylabel('Amplitude (mV)');
% 
% subplot(N+2,1,N+2);
% plot(time_in_ms, clip_artifact_downsampled/1e3);
% title('artifact template component');
% xlabel('Time (ms)');
% ylabel('Amplitude (mV)');

% Create the bar chart for singular values
figure;
bar(singular_values);
title('singular values in descending order');
xlabel('Index');
ylabel('Value');