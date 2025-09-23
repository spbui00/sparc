% SCRIPT TO PLOT **ONLY THE ARTIFACT**
clear;
clc;
data_file_path = '/Users/bui/code/sparc/research/generate_dataset/SimulatedData.mat';
RawDataSampleRate = 30000;

% --- Compare two different electrodes ---
electrode_1 = 1; % The stimulating electrode
electrode_2 = 2; % A nearby electrode
array_to_plot = 1;

load(data_file_path);
num_samples = size(SimArtifact, 1);
time_vector = (0:num_samples-1) / RawDataSampleRate;

figure;

% --- Top Plot: Artifact at the Source ---
subplot(2, 1, 1);
plot(time_vector, SimArtifact(:, electrode_1, array_to_plot));
title_string = sprintf('ARTIFACT ONLY on Stimulating Electrode (%d, %d)', array_to_plot, electrode_1);
title(title_string);
ylabel('Amplitude (\muV)');
grid on;
xlim([0 0.2]);

% --- Bottom Plot: Artifact on a Different Electrode ---
subplot(2, 1, 2);
plot(time_vector, SimArtifact(:, electrode_2, array_to_plot));
title_string = sprintf('ARTIFACT ONLY on Nearby Electrode (%d, %d)', array_to_plot, electrode_2);
title(title_string);
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
grid on;
xlim([0 0.2]);

disp('Plot generated successfully.');