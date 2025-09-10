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

select_clip_num = 1;
Ain_t = squeeze(data_in(select_clip_num,:,:));
% Create time vector in milliseconds
sampling_rate = 512; 
time_in_ms = (0:size(data_in, 2)-1) / sampling_rate * 1000; % Convert to milliseconds




%% template subtraction
%hyper parameters
% template_length_ms = 380; % in millisecond
% template_length_timesteps = floor(template_length_ms  / 1e3 * sampling_rate);
template_length_timesteps = 90; % 3ms stimulation rate
stimulation_start_timestep = 1513;
num_of_templates_for_avg = 3;

total_timesteps = size(Ain_t, 1);
n_cycle = floor((total_timesteps - stimulation_start_timestep) / template_length_timesteps);
residual = (total_timesteps - stimulation_start_timestep) - n_cycle * template_length_timesteps;

channel_num = size(Ain_t,2);
Dout_clean = zeros(total_timesteps, channel_num);

for ch = 1:channel_num
    Ain = Ain_t(:,ch);
    Ain = Ain';
    Dout=Ain;  
    % Extract and subtract average template
    for t = stimulation_start_timestep:template_length_timesteps:(n_cycle - num_of_templates_for_avg) * template_length_timesteps
        
        % Extract the last k templates
        templates = zeros(num_of_templates_for_avg, template_length_timesteps);
        for k = 1:num_of_templates_for_avg
            templates(k,:) = Ain(t+(k-1)*template_length_timesteps:t+k*template_length_timesteps-1);
        end
        % Compute the average template
        avg_template = mean(templates, 1);
        
        % Subtract average template from the current window
        Dout(t:t+template_length_timesteps-1) = Ain(t:t+template_length_timesteps-1) - avg_template;
        
    end
    % subtract template for the remaining k-1 cycles and the residual (if
    % residual >0)
    for k = 1:num_of_templates_for_avg
        Dout(t+k*template_length_timesteps:t+(k+1)*template_length_timesteps-1) = Ain(t+k*template_length_timesteps:t+(k+1)*template_length_timesteps-1) - avg_template;
    end
    if residual >0
        Dout(t+(k+1)*template_length_timesteps:end) = Ain(t+(k+1)*template_length_timesteps:end) - avg_template(1:residual);
    end


    Dout_clean(:, ch) = Dout;

end


%% plot
selected_channel_number = 1;
figure();
plot(time_in_ms, squeeze(data_in(select_clip_num,:,selected_channel_number))/1e3);
hold on;
plot(time_in_ms, Dout_clean(:,selected_channel_number)/1e3);
hold off;
legend('original signal','reconstructed signal')
grid on;
xlabel('Time (ms)');
ylabel('Voltage (mV)');





