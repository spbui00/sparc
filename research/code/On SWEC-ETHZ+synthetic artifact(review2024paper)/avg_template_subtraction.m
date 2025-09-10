clear;
clc;
close all;


%% SWEC-ETHZ iEEG dataset
seizure_flag = true;
data_folder = 'high_amp_73/';
if seizure_flag
    load(fullfile(data_folder, 'swec-ethz-ieeg-seizure-data-rate512Hz.mat'));
    data_in = mixed_seizure;
    synthetic_GT = signal_seizure;
else
    load(fullfile(data_folder, 'swec-ethz-ieeg-nonseizure-data-rate512Hz.mat'));
    data_in = mixed_nonseizure;
    synthetic_GT = signal_nonseizure;
end

data_in = permute(data_in, [1,3,2]); %convert to [trials, timesteps, channels]
synthetic_GT = permute(synthetic_GT, [1,3,2]); %convert to [trials, timesteps, channels]

select_clip_num = 1;
Ain_t = squeeze(data_in(select_clip_num,:,:));
synthetic_GT = squeeze(synthetic_GT(select_clip_num,:,:));
% Create time vector in milliseconds
sampling_rate = 512; 
time_in_ms = (0:size(data_in, 2)-1) / sampling_rate * 1000; % Convert to milliseconds




%% template subtraction
%hyper parameters
% template_length_ms = 380; % in millisecond
% template_length_timesteps = floor(template_length_ms  / 1e3 * sampling_rate);
template_length_timesteps = 200;
num_of_templates_for_avg = 3;

total_timesteps = size(Ain_t, 1);
n_cycle = floor(total_timesteps / template_length_timesteps);
residual = total_timesteps - n_cycle * template_length_timesteps;

channel_num = size(Ain_t,2);
Dout_clean = zeros(total_timesteps, channel_num);

for ch = 1:channel_num
    Ain = Ain_t(:,ch);
    Ain = Ain';
    Dout=Ain;  
    % Extract and subtract average template
    for t = 1:template_length_timesteps:(n_cycle - num_of_templates_for_avg) * template_length_timesteps
        
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
if seizure_flag
    save('/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/ethz_data/interp/OldData_avgtemp_seizure_amp73.mat', 'Dout_clean');
else
    save('/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/ethz_data/interp/OldData_avgtemp_nonseizure_amp73.mat', 'Dout_clean');
end
% Calculate and display some comparison metrics
[mse, psd] = SynGT_performance_metrics(synthetic_GT, Dout_clean);

%% plot
selected_channel_number = 1;
figure();
plot(time_in_ms, Dout_clean(:,selected_channel_number)/1e3,  'LineWidth', 2, 'Color',[1, 0, 0] ); %light gray:[0.8, 0.8, 0.8]
hold on;
plot(time_in_ms, synthetic_GT( :,selected_channel_number)/1e3, LineWidth=2.5, Color='blue');
hold off;
grid on;
ylim([-0.5, 0.5])
xlim([0 4000])
legend('Apply avg template subtraction', 'GT seizure signal','FontSize', 16);
xlabel('Time (ms)', 'FontSize', 16);
ylabel('Voltage (mV)', 'FontSize', 16);
set(gca, 'FontSize', 16);  % Set font size for tick labels





