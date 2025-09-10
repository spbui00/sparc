%% Load test dataset
clear
clear variables; 
clc
close all;

%% SWEC-ETHZ iEEG dataset
seizure_flag = false;
data_folder = 'high_amp_70/';
if seizure_flag
    load(fullfile(data_folder, 'swec-ethz-ieeg-seizure-data-rate512Hz.mat'));
    data_in = mixed_seizure;
    synthetic_GT = signal_seizure;
else
    load(fullfile(data_folder, 'swec-ethz-ieeg-nonseizure-data-rate512Hz.mat'));
    data_in = mixed_nonseizure;
    synthetic_GT = signal_nonseizure;
end

dataInt = permute(data_in, [1,3,2]); %convert to [trials, timesteps, channels]
synthetic_GT = permute(synthetic_GT, [1,3,2]); %convert to [trials, timesteps, channels]

select_clip_num = 1;
synthetic_GT = squeeze(synthetic_GT(select_clip_num,:,:));
% Create time vector in milliseconds
fs_data = 512; 
time_in_ms = (0:size(dataInt, 2)-1) / fs_data * 1000; % Convert to milliseconds



%% Setup ERAASR Parameters
addpath('/home/ni/Documents/artifact-cancellation/datasets/eraasr-1.0.0/')
opts = ERAASR.Parameters();
opts.Fs = fs_data; % samples per second
Fms = opts.Fs / 1000; % multiply to convert ms to samples

opts.thresholdHPCornerHz = 250;
opts.thresholdChannel = 1;
opts.thresholdValue = -1e-4;

opts.alignChannel = 1;
opts.alignUpsampleBy = 10;
opts.alignWindowPre = round(Fms * 0.5);
opts.alignWindowDuration = round(Fms * 12);

% 30 ms stim, align using 10 ms pre start to 55 post 
% scale proportionally according to ERASSR's original parameters
opts.extractWindowPre = round(Fms * 0);
opts.extractWindowDuration = round(Fms * 55);
opts.cleanStartSamplesPreThreshold = round(Fms * 0.5);
        
opts.cleanHPCornerHz = 10; % light high pass filtering at the start of cleaning
opts.cleanHPOrder = 4; % high pass filter order 
opts.cleanUpsampleBy = 1; % upsample by this ratio during cleaning
opts.samplesPerPulse = round(Fms * 3); % 3 ms pulses
opts.nPulses = 20;

opts.nPC_channels = 12;
opts.nPC_trials = 2;
opts.nPC_pulses = 6;

opts.omit_bandwidth_channels = 3;
opts.omit_bandwidth_trials = 1;
opts.omit_bandwidth_pulses = 1;

opts.alignPulsesOverTrain = false; % do a secondary alignment within each train, in case you think there is pulse to pulse jitter. Works best with upsampling
opts.pcaOnlyOmitted = true; % if true, build PCs only from non-omitted channels/trials/pulses. if false, build PCs from all but set coefficients to zero post hoc

opts.cleanOverChannelsIndividualTrials = false;
opts.cleanOverPulsesIndividualChannels = false;
opts.cleanOverTrialsIndividualChannels = false;

opts.cleanPostStim = true; % clean the post stim window using a single PCR over channels

opts.showFigures = false; % useful for debugging and seeing well how the cleaning works
opts.plotTrials = 1; % which trials to plot in figures, can be vector
opts.plotPulses = 1; % which pulses to plot in figures, can be vector
opts.figurePath = pwd; % folder to save the figures
opts.saveFigures = false; % whether to save the figures
opts.saveFigureCommand = @(filepath) print('-dpng', '-r300', [filepath '.png']); % specify a custom command to save the figure

%% Do alignment and cleaning procedure

[dataCleaned, extract] = ERAASR.cleanTrials(dataInt, opts);


%% Note before spike extraction
% It would presumably make sense to combine the stimulated trials with any 
% non-stimulated trials to ensure that the broadband signals are treated
% identically from this point forward

%% High pass filter the cleaned data

dataCleanedHP = ERAASR.highPassFilter(dataCleaned, opts.Fs, 'cornerHz', 250, 'order', 4, ...
    'subtractFirstSample', true, 'filtfilt', false, 'showProgress', true);
        
%% Spike thresholding and waveform extraction

rmsThresh = -4.5 * ERAASR.computeRMS(dataCleanedHP, 'perTrial', false, 'clip', 60); % clip samples that sink outside +/- 60 uV

waveSamplesPrePost = [10 38];
[spikeTimes, waveforms] = ERAASR.extractSpikesCrossingThreshold(dataCleanedHP, rmsThresh, ...
    'mode', 'largestFirst', 'waveformSamplesPrePost', waveSamplesPrePost, 'lockoutPrePost', [9 30]);

%% Plot the mean spike waveforms
% Note that this will include some spikes outside of the stimulation period as is

nChannels = size(waveforms, 2);
nSamples = sum(waveSamplesPrePost);
waveformMeans = nan(nSamples, nChannels);
for iC = 1:nChannels
    waveformMeans(:, iC) = mean(cat(1, waveforms{:, iC}), 1);
end

figure();
plot(waveformMeans);
box off;

% Calculate and display some comparison metrics
[mse] = SynGT_performance_metrics(synthetic_GT, squeeze(dataCleaned(select_clip_num,:,:)));
%% Plot results
selected_channel_number = 1;

figure();
plot(time_in_ms, synthetic_GT(:,selected_channel_number));
hold on;
plot(time_in_ms, squeeze(dataCleaned(select_clip_num,:,selected_channel_number)));
hold off;

legend('GT', 'Reconstructed signal');
xlabel('Time (ms)');