% clear;
% clear variables; 
% close all;
% clc;

% hyperparameters
N = 10; % Replace with the desired number of largest singular values to zero out
cut_window = 1500:5000; % cut window for comparison
benchmark_window = 1:1500; % benchmark window
fs = 30000; % Sampling frequency (30 kHz)

% original data load 
dat=load('exampleDataTensor.mat');
data_in=dat.data_trials_by_time_by_channels;
Ain_t=squeeze(data_in(1,:,:));
% % Rescale to [-1,1] and store the original min and max for later use
% original_min = min(Ain_t(:));
% original_max = max(Ain_t(:));
% Ain_nt = -1 + 2.*(Ain_t - min(Ain_t))./(max(Ain_t) - min(Ain_t));

% synthetic data load
dat=load('SyntheticNeuralSig.mat');
synthetic_GT = dat.synthetic_neural_signal;
synthetic_GT = squeeze(synthetic_GT(1,:,:));

figure();
plot(synthetic_GT(:, 1));
hold on;
plot(Ain_t(:,1));
hold off;
legend('synthetic','real')

% Perform SVD 
[U, S, V] = svd(Ain_t);

% Sort singular values in descending order and get their indices
[sorted_values, sort_indices] = sort(diag(S), 'descend');

% Determine which indices correspond to the N largest values
largest_N_indices = sort_indices(1:N);

% Create a copy of S with the N largest singular values zeroed out
S_filtered = S;
S_filtered(largest_N_indices, largest_N_indices) = 0;

% Reconstruct the signal with removed components
reconstructed_signal = U * S_filtered * V';

% Reverse the scaling to compare with the original signal
% reconstructed_signal_rescaled = (reconstructed_signal + 1) / 2 * (original_max - original_min) + original_min;
% % Calculate the offset based on the difference at the first timestep
% offset = Ain_t(1,:) - reconstructed_signal_rescaled(1,:);
% % Add the offset to the reconstructed signal
% reconstructed_signal_rescaled = reconstructed_signal_rescaled + offset;

%% Calculate and display some comparison metrics
[mse, psd_mse_values] = SynGT_performance_metrics(synthetic_GT, reconstructed_signal);

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



figure();
plot(squeeze(synthetic_GT(:,1)));
hold on;
plot(squeeze(Ain_t(:,1)));
hold on;
plot(squeeze(reconstructed_signal(:, 1)));
hold off;
legend('synthetic GT without artifacts','real signal with artifacts','reconstructed signal')
grid on;



figure;
% Calculate the removed signal
removed_signal = Ain_t - reconstructed_signal;
% Plot the original signal, reconstructed signal, and removed signal
t = 1:length(Ain_t);
subplot(3,1,1);
plot(t, Ain_t(:,1));
title('Original Signal');
xlabel('Time');
ylabel('Amplitude');

subplot(3,1,2);
plot(t, reconstructed_signal(:,1));
title('Reconstructed Signal (Noise Removed)');
xlabel('Time');
ylabel('Amplitude');

subplot(3,1,3);
plot(t, removed_signal(:,1));
title('Removed Signal (Potential Noise)');
xlabel('Time');
ylabel('Amplitude');

% Create the bar chart for singular values
figure;
bar(sorted_values);
title('singular values in descending order');
xlabel('Index');
ylabel('Value');