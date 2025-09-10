function [mse, psd_mse] = SynGT_performance_metrics_allTrials(synthetic_GT, reconstructed_signal)
    % Calculate performance metrics (MSE and PSD MSE) for neural signals
    %
    % Inputs:
    %   synthetic_GT: Ground truth synthetic signal 
    %   [trial, timestep, channel]
    %   reconstructed_signal: Reconstructed or compared signal
    %
    % Outputs:
    %   mse: Mean Squared Error between the signals
    %   psd_mse_values: Mean Squared Error between the PSDs for each channel

    % Hyperparameters
    cut_window = 1:512; % cut window for comparison
    fs = 512; % Sampling frequency 
    nfft = 256; % Number of FFT points, adjust as needed
    noverlap = nfft/2; % 50% overlap

    % Ensure the signals have the same number of channels
    num_channels = min(size(synthetic_GT, 3), size(reconstructed_signal, 3));

    % Calculate MSE
    mse_values = (synthetic_GT(:, cut_window, 1:num_channels) - reconstructed_signal(:, cut_window, 1:num_channels)).^2;

    mse = mean(mse_values, 'all');

    % Calculate PSD MSE
    % Assume: synthetic_GT and reconstructed_signal are [trials, timesteps, channels]
    [trial_num, ~, ~] = size(synthetic_GT);
    psd_mse_values = zeros(trial_num, num_channels);
    
    for trial = 1:trial_num
        for ch = 1:num_channels
            % Extract time series for this trial and channel
            signal_synthetic = squeeze(synthetic_GT(trial, cut_window, ch));
            signal_reconstructed = squeeze(reconstructed_signal(trial, cut_window, ch));
    
            % Compute PSD using pwelch
            [psd_synthetic, ~] = pwelch(signal_synthetic, hanning(nfft), noverlap, nfft, fs);
            [psd_reconstructed, ~] = pwelch(signal_reconstructed, hanning(nfft), noverlap, nfft, fs);
    
            % Compute MSE of PSDs
            psd_mse_values(trial, ch) = mean((psd_synthetic - psd_reconstructed).^2);
        end
    end

    % Average over all trials and channels
    psd_mse = mean(psd_mse_values, 'all');


    % Display results
    fprintf("MSE:\n");
    fprintf('%f\n', mse);
    fprintf("\nPSD MSE values:\n");
    fprintf('%f\n', psd_mse);
end