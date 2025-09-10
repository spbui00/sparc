function [mse, psd_mse] = SynGT_performance_metrics(synthetic_GT, reconstructed_signal)
    % Calculate performance metrics (MSE and PSD MSE) for neural signals
    %
    % Inputs:
    %   synthetic_GT: Ground truth synthetic signal
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
    num_channels = min(size(synthetic_GT, 2), size(reconstructed_signal, 2));

    % Calculate MSE
    mse = mean(mean((synthetic_GT(cut_window, 1:num_channels) - reconstructed_signal(cut_window, 1:num_channels)).^2));

    % Calculate PSD MSE
    psd_mse_values = zeros(num_channels, 1);
    for i = 1:num_channels
        [psd_synthetic, ~] = pwelch(synthetic_GT(cut_window, i), hanning(nfft), noverlap, nfft, fs);
        [psd_reconstructed, ~] = pwelch(reconstructed_signal(cut_window, i), hanning(nfft), noverlap, nfft, fs);
        psd_mse_values(i) = mean((psd_synthetic - psd_reconstructed).^2);
    end
    psd_mse = mean(psd_mse_values);

    % Display results
    fprintf("MSE:\n");
    fprintf('%f\n', mse);
    fprintf("\nPSD MSE values:\n");
    fprintf('%f\n', psd_mse);
end