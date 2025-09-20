% SCRIPT TO PLOT SIMULATED NEURAL DATA

clear; % Clears the workspace to avoid conflicts
clc;   % Clears the command window

% --- Configuration ---
% Define the full path to your data file
data_file_path = '/Users/bui/code/sparc/research/generate_dataset/SimulatedData_2x64_30000.mat';
RawDataSampleRate = 30000;

% Define which channel to plot
electrode_to_plot = 1;
array_to_plot = 1;

% --- Main Code ---
% Load your generated data file
load(data_file_path);

% Create a time vector in seconds for the x-axis
num_samples = size(SimCombined, 1);
time_vector = (0:num_samples-1) / RawDataSampleRate;

% Create the plot
figure;
plot(time_vector, SimCombined(:, electrode_to_plot, array_to_plot));

% Add labels and a title for clarity
title_string = sprintf('Combined Signal for Array %d, Electrode %d', array_to_plot, electrode_to_plot);
title(title_string);
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
grid on;

% Zoom in to see the details of the waveform (e.g., the first 200ms)
xlim([0 0.2]);

disp('Plot generated successfully.');