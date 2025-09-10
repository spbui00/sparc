clear;
clc;
close all;


%% load stanford dataset
% load original data
dat=load('exampleDataTensor.mat');
data_in=dat.data_trials_by_time_by_channels;
total_trials = size(data_in,1);
total_channels = size(data_in,3);
num_timesteps = size(data_in,2);

% Create time vector in milliseconds
sampling_rate = 512; 
time_in_ms = (0:size(data_in, 2)-1) / sampling_rate * 1000; % Convert to milliseconds

% plot
selected_trial_number = 1;
selected_channel_number = 1;
% Create main figure
fig = figure();
plot(data_in(selected_trial_number,:,selected_channel_number));
grid on;

%% template subtraction
%hyper parameters
% template_length_ms = 380; % in millisecond
% template_length_timesteps = floor(template_length_ms  / 1e3 * sampling_rate);
template_length_timesteps = 90;
num_of_templates_for_avg = 3;
cut_window = 1549:3348;

total_timesteps_artifact = size(cut_window, 2);
n_cycle = total_timesteps_artifact / template_length_timesteps;


Dout_clean = zeros(total_trials, num_timesteps, total_channels);

for trial_num = 1:total_trials
    Ain_t=squeeze(data_in(trial_num,:,:));

    for ch = 1:total_channels
        Ain = Ain_t(:,ch);
        Ain = Ain';
        Dout=Ain;  
        % Extract and subtract average template
        for t = cut_window(1):template_length_timesteps:cut_window(1)+ (n_cycle - num_of_templates_for_avg) * template_length_timesteps
            
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
        for k = 1:(num_of_templates_for_avg-1)
            Dout(t+k*template_length_timesteps:t+(k+1)*template_length_timesteps-1) = Ain(t+k*template_length_timesteps:t+(k+1)*template_length_timesteps-1) - avg_template;
        end
    
    
        Dout_clean(trial_num, :, ch) = Dout;
    
    end
end



%% plot
selected_trial_number = 1;
selected_channel_number = 1;
figure();
plot(time_in_ms, Dout_clean(selected_trial_number,:,selected_channel_number)/1e3);
grid on;
xlabel('Time (ms)');
ylabel('Voltage (mV)');





