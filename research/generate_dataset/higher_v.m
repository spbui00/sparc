% this script is a cleaned version from the artifact removal paper
% It is a simple script that takes the geometry of the Utah arrays in
% Macaque V1. Assuming there is a single, distant ground and reference,
% simulate the neural response from microstimulation from a single
% electrode.
% First, construct the electrode geometry, randomize a baseline firing
% rate (FR)
% and local field potential (LFP) for every electrode
% Next, define stimulation effect on instantaneous FR using a simple DOG model. Assuming there
% is an increased FR on the electodes close to the
% stimulating electrode and a decreased FR on the electrodes in a band a
% bit farther away from the stimuleting electrode. For the purpose of the
% investigation (artifact removal) the accuracy of this profile is not
% critical. We use a Poisson spike generator to generate the spike timing.
% We also use a randomly predefined spike waveform to generate spike
% wavefoms
% Next, simulate a spike triggered LFP, based on the spike timing, to generate a LFP response to
% electrical stimulation. Again, the accuracy of this profile is not
% critical for the purpose of the investigation. All parameters are open for tuning.
% Next, we need to acurately simulate the stimulation artifacts, there are two
% ways of generating simulated artifacts. The first one is to generate the
% stimulation waveform and filter the input. The second is to use averaged
% prestored stimulation artifact from measurement (Assuming you have
% measurement).
% In the end, we simply sum all the components together to form the
% response without artifact and response with artifact.

clc, clear, close all
ElectrodeImpedance = 150; % kOhm, 150 kOhm is the representative value of Utah arrays with IOx coating. According to Ohm's law, the stimulation artifact amplitude should be proportional to the current level.
TissueImpedanceperMM = 6; % kOhm/mm
TissueCapacitanceperMM = 1.5; % nano(10^-9) farad
ElectrodeCapacitance = 1; % nano(10^-9) farad
FRreductionRatio = 8; % This is a simple parameter allows you to reduce the FR
FrameRate = 25; % camera frames per second. Assuming the stimulator is connected to a camera, the camera image will dictate the stimulation strength.
SimulationLength = 4; % The total simulated data length in seconds.
FrameTime = 1/FrameRate;
NumFrames = SimulationLength/FrameTime;
PhospheneThreshold = 100; % This is the arbitrary phosphene threshold of the stimulating electrode, it is used in calculate stimulation induced response
MaxCurrentLevelMultiplier = 3; % This is the maximal stimulation current level on a channel. It is PhospheneThreshold * MaxCurrentLevelMultiplier
GreyLevels = 8; % assume we can control the gray level, but here is more relevant in testing artifact removing
StimParam.Frequency = 50; % pulse frequency Hz
StimParam.Period = 1/StimParam.Frequency; % in seconds
StimParam.timeperphase = 0.17; % ms, same as cerestim definition, this is the time per phase in a single stim pulse
StimParam.timeinterphase = 0.06; % ms,  same as cerestim definition, this is the time between phases in a single stim pulse
StimParam.CurrentLevelFerFrame = randi(GreyLevels,NumFrames,1) / GreyLevels * PhospheneThreshold * MaxCurrentLevelMultiplier; % this is the stimulation current level per frame
SpikeWaveformSNR = 3;
ArtifactNoiseLevel = 0.01; % 10% artifact noise level
ArtifactLevel100uA = 50000; % uV
RawDataSampleRate = 30000;
SampleTime = 1/RawDataSampleRate;

NumRecordingArray = 2; % the recording array index. Define how many recording arrays to be simulated
RecordingElecPerArray = 64; % the recording electrode index

SpatialActivationThreshold = 0.01; % For every recording electrode, we simply assume a range of stimulation induced activity on the cortex that following a profile (e.g. 2D Gaussian) when the profile value is lower than this threshold, there is no effect of stimulation on this electrode.
ActivationThreshold = 0.01;

StimulatedArray = 1:16:16; % define the simulated arrays, let's simulate only one array as artifact removal from other arrays are almost trivial
StimulatingElectrode = 1; % define the electrode ID on this one simulated array, feel free the change how many you want to simulate
StimulationExcitationSigmaMean = [1 0.5]; % ms
StimulationExcitationCoefficient = 1;
StimulationInhibitionSigmaMean = [5 10]; % ms
StimulationInhibitionCoefficient = 0.1;
% StimulationSpatialActivationSigmaMean = [0.5 0; 1.5,0]; % in mm
StimulationSpatialActivationSigmaMean = [2 0; 1.5,0];
StimulationSpatialCoefficient = [1, 0];

LFPspatialSigmaMean = [0.15 0]; % mm
% SpikeTriggeredLFP parameters are inline, see code below


ArtifactSimulationMethod = 1; % stimulation artifact simulation method. 1 = hand crafted, 2 = RC circuit simulation (very slow), 3 = from real data

minISI = 0.1; % ms, MUA
FRscaleFactor = 3; % MUA

plotArtifact = false;

% spike detection bandpass filter
Fn = RawDataSampleRate/2;
Fbp=[250,5000];
N  = 2;    % filter order
[Bbp, Abp] = butter(N, [min(Fbp)/Fn max(Fbp)/Fn],'bandpass'); % BandPass
[Bhigh, Ahigh] = butter(N, min(Fbp)/Fn,'high'); % BandPass

% load pregenerated spikewaveforms, example random waveforms, please
% replace with better simulation or real recorded waveforms.
load('./AllSpikeWaveform')

% load example array configuration
load('./ArrayParam.mat')


% build frame stim timing info
LastTimeStampIdx = 0;
AllStimIdx = [];
for thisFrame = 1:NumFrames
    FrameStartTime = (thisFrame-1) * FrameTime;
    % time stamps
    timeStamps = 0:SampleTime:(FrameTime-SampleTime);
    idx = 1:numel(timeStamps);
    idx = idx + LastTimeStampIdx;
    % stimulation time stamps
    StimTime = 0:StimParam.Period:(FrameTime-StimParam.Period);
    StimIdx = round(StimTime/SampleTime)+1;
    StimIdx = StimIdx + LastTimeStampIdx;
    LastTimeStampIdx = idx(end);
    AllStimIdx = [AllStimIdx, StimIdx];
end

% build electrode data tensor
% build electrode spike train
SimFR = zeros(LastTimeStampIdx/RawDataSampleRate*1000,RecordingElecPerArray,NumRecordingArray);
SimSpikeTrain = zeros(ceil(NumFrames*FrameTime*RawDataSampleRate),RecordingElecPerArray,NumRecordingArray); % ms
SimLFP = zeros(LastTimeStampIdx,RecordingElecPerArray,NumRecordingArray);
SimBB = zeros(LastTimeStampIdx,RecordingElecPerArray,NumRecordingArray);
SimArtifact = zeros(LastTimeStampIdx,RecordingElecPerArray,NumRecordingArray);

initialized = false;
for StimArray = StimulatedArray % for each stimulating array
    for StimElec = StimulatingElectrode % for each stimulating electrode on the array. Please revise if you want to simulate stimulating from different electrode on multiple arrays
        disp(['Array=',num2str(StimArray),' Elec=',num2str(StimElec)])
        % simulation instantaneous FR
        % For every recording electrode, the instantaneous FR is simply
        % simulated as a baseline FR plus the stimulation induced FR
        fprintf('Simulating fring rate...\n')
        for thisArray = 1:NumRecordingArray
            for thisElectrode = 1:RecordingElecPerArray

                % for every recording electrode, generate a random stimulation induced effect kernal function
                % using DOG model for simplicity (not important for artifact
                % removal investigation)
                x = 0:0.01:30; % ms, use fixed kernal length here, time resolution is 1/100 ms
                x = x/RawDataSampleRate*1000;
                y1 = gaussmf(x,StimulationExcitationSigmaMean/RawDataSampleRate*1000);
                y2 = -gaussmf(x,StimulationInhibitionSigmaMean/RawDataSampleRate*1000);
                Kstim = StimulationExcitationCoefficient*y1 + StimulationInhibitionCoefficient*y2;
                x1 = ArrayParam(thisArray).Xmmcoord(thisElectrode);
                y1 = ArrayParam(thisArray).Ymmcoord(thisElectrode);
                x2 = ArrayParam(StimArray).Xmmcoord(StimElec);
                y2 = ArrayParam(StimArray).Ymmcoord(StimElec);
                Distance = ((x1-x2)^2 + (y1-y2)^2)^0.5; % distance between the recording and stimulating electrode
                
                % for the sitimulating electrode
                if Distance < 0.1
                    Distance = 0.1;
                end

                SpatialActivationFactor = StimulationSpatialCoefficient(1)*gaussmf(Distance,StimulationSpatialActivationSigmaMean(1,:)) - StimulationSpatialCoefficient(2)*gaussmf(Distance,StimulationSpatialActivationSigmaMean(2,:)); % spatial activation factor scales the stimulation induced response based on the distance between the stimulating and recording electrode.

                if initialized % initialize baseline FR
                    FR = zeros(ceil(NumFrames*FrameTime*1000)*100,1,1);
                else
                    FR = ones(ceil(NumFrames*FrameTime*1000)*100,1,1); % model firing rate a the resolution of 1/100 ms
                    FR = FR * ArrayParam(thisArray).baselineFR(thisElectrode); % baseline FR
                end
                % stim FR
                % Kstim(stimulating electrode) * activation factor
                % activation factor follows the logistic response function of the stimulating
                % electrode centered on threshold
                % f(x) = L/(1+e^-k(x-x0))
                % L: max value
                % k: growth rate
                % x0: the x value of the sigmoid midpoint
                LastTimeStampIdx = 0;
                FRmod0 = [];
                for thisFrame = 1:NumFrames
                    k = ArrayParam(StimArray).k(StimElec);
                    x0 = ArrayParam(StimArray).threshold(StimElec);
                    x = StimParam.CurrentLevelFerFrame(thisFrame);
                    ActivationFactor = 1 ./ (1 + exp(-k*(x-x0)));
                    if abs(SpatialActivationFactor) > SpatialActivationThreshold && abs(ActivationFactor) > ActivationThreshold
                        Kstim0 = Kstim * ActivationFactor * SpatialActivationFactor*(ArrayParam(thisArray).MaxFR(thisElectrode) - ArrayParam(thisArray).baselineFR(thisElectrode)); % the induced change should be a fraction of the max FR
                        timeStamps = 0:0.01:(FrameTime*1000-0.01); % 1/100 ms time resolution
                        % stimulation time stamps
                        StimTime = (0:StimParam.Period:(FrameTime-StimParam.Period))*1000;
                        StimIdx = round(StimTime/0.01)+1;
                        StimTrain = zeros(size(timeStamps));
                        StimTrain(StimIdx) = 1;
                        FRmod = conv(StimTrain,Kstim0);
                        FRmod = FRmod(1:numel(timeStamps));
                        idx = 1:numel(timeStamps);
                        idx = idx + LastTimeStampIdx;
                        LastTimeStampIdx = idx(end);
                        % FR will be a simple linear summation between the
                        % baseline FR and the induced FR
                        if idx(1)+ length(FRmod) - 1 <= length(FR)
                            FR(idx(1):(idx(1)+ length(FRmod) - 1)) = FR(idx(1):(idx(1)+ length(FRmod) - 1)) + FRmod';
                        else
                            FR(idx(1):end) = FR(idx(1):end) + FRmod(1:(length(FR)-(idx(1))+1))';
                        end
                    end
                    idx = FR > 1000;
                    FR(idx) = 1000;
                    idx = FR < 0;
                    FR(idx) = 0;
                end
                % down sample FR to 1ms, unit is in Hz
                FR = downsample(FR,100);
                % unit conversion
                FR = FR/1000*FRscaleFactor;
                SimFR(:,thisElectrode,thisArray) = FR;
                % Simple Poisson spike generator
                randNumber = random('Uniform',0,1,size(FR,1),size(FR,2));
                SpikeTrain = double((randNumber - FR) <= 0);
                SpikeTrain = upsample(SpikeTrain,RawDataSampleRate/1000);
                SpikeTime = find(SpikeTrain==1);
                idx = SpikeTime < 1;
                SpikeTime(idx) = [];
                idx = SpikeTime > length(SpikeTrain);
                SpikeTime(idx) = [];
                SpikeTrain = zeros(size(SpikeTrain));
                SpikeTrain(SpikeTime) = 1;
                SpikeTrain = double(logical(SpikeTrain) | logical(SimSpikeTrain(:,thisElectrode,thisArray)));
                % ISI
                SpikeTime = find(SpikeTrain > 0);
                SpikeTimeDiff = diff(SpikeTime);
                idx = SpikeTimeDiff <= minISI * RawDataSampleRate/1000;
                SpikeTime(idx+1) = [];
                SpikeTrain = zeros(size(SpikeTrain));
                SpikeTrain(SpikeTime) = 1;
                % end ISI
                SimSpikeTrain(:,thisElectrode,thisArray) = SpikeTrain; % store the spike train in RawDataSampleRate
            end
        end % end of spike train simulation

        % stimultion artifact simulation
        fprintf('Simulating artifacts...\n')
        for thisArray = 1:NumRecordingArray
            for thisElectrode = 1:RecordingElecPerArray
                switch ArtifactSimulationMethod
                    case 1 % hand crafted artifact
                        % hand crafted artifact, good for testing
                        withinframesizevar = 0.2;
                        amplitudeshiftingintime = 0.12;
                        beta0 = [300.0, 0.5];
                        % artifact spatial profile
                        x1 = ArrayParam(thisArray).Xmmcoord(thisElectrode);
                        y1 = ArrayParam(thisArray).Ymmcoord(thisElectrode);
                        x2 = ArrayParam(StimArray).Xmmcoord(StimElec);
                        y2 = ArrayParam(StimArray).Ymmcoord(StimElec);
                        Distance = ((x1-x2)^2 + (y1-y2)^2)^0.5;
                        % for the sitimulating electrode
                        if Distance < 0.1
                            Distance = 0.1;
                        end
                        LastTimeStampIdx = 0;
                        for thisFrame = 1:NumFrames
                            currentLevel = StimParam.CurrentLevelFerFrame(thisFrame);
                            % yy = @(b,x) b(1) .*currentLevel.* (b(2).*x).^(-2);
                            % yy2 = @(b,x) b(1) .*10.* (b(2).*x).^(-2);
                            yy = @(b,x) b(1) .*currentLevel.* (b(2).*x).^(-0.5);
                            yy2 = @(b,x) 1 .*10.* (b(2).*x).^(-1);
                            r1 = random('Normal',beta0(1),beta0(1)/100);
                            r2 = random('Normal',beta0(2),beta0(2)/100);
                            beta(1) = r1;
                            beta(2) = r2;
                            SpatialArtifactFactor = yy(beta,Distance);
                            idx = 1:(FrameTime/SampleTime);
                            idx = idx + LastTimeStampIdx;
                            LastTimeStampIdx = idx(end);
                            if SpatialArtifactFactor > 0.001

                                % stimulation artifact kernel as a sin wave
                                % followed by a expoential decay to
                                % simulate capacitor discharge
                                n = round(random('Uniform',4, 6));
                                x = 0:(1/30):15; % ms
                                y1 = -sin(2*pi/x((n-1)*4)*(x(1:((n-1)*4))));
                                r3 = abs(random('Normal',0.4,0.04));
                                % y2 should be exponential decay with the following
                                lamda1 = 1.5 - 1.5/2 * ((currentLevel/50)/(210/50));
                                lamda2 = 3 - 3/2 * ((currentLevel/50)/(210/50));
                                y2 = (exp(-lamda1*(x-x(n+1)))+exp(-lamda2*(x-x(n+1))));
                                y2 = y2/max(y2);
                                y = (y1(n)/abs(y1(n)))*ones(size(y2));
                                y(1:numel(y1)) = y1;
                                y((n+1):end) = y((n+1):end).*y2(n+1:end);
                                Kart = y;

                                Kart0 = Kart * SpatialArtifactFactor/yy2(beta,0.3);
                                % stimulation time stamps
                                StimTime = (0:StimParam.Period:(FrameTime-StimParam.Period))*1000;
                                StimIdx = round(StimTime/(1/30))+2;
                                StimTrain = zeros(1,FrameTime/SampleTime);
                                % randomize size within frame
                                for thisStimPulse = 1:numel(StimIdx)
                                    StimTrain(StimIdx(thisStimPulse)) = random('Uniform',1-withinframesizevar/2,1+withinframesizevar/2);
                                end
                                % amplitude shifting in time
                                time0 = (thisFrame-1) * FrameTime;
                                time1 = thisFrame * FrameTime;
                                timeTicks = linspace(time0,time1,size(StimTrain,2));
                                signnum = rand(1)-0.5;
                                if signnum == 0
                                    signnum = 1;
                                else
                                    signnum = signnum/abs(signnum);
                                end
                                AmpTimeShift = signnum.*amplitudeshiftingintime.*sin(2.*pi.*(timeTicks/(rand(1)*(1)+0.1)) + rand(1).*2.*pi);
                                AmpTimeShift = AmpTimeShift + (1-AmpTimeShift(1)); % make sure AmptimeShit start with 1
                                StimTrain = StimTrain.*AmpTimeShift;

                                StimArtifact = conv(StimTrain,Kart0);
                                StimArtifact = StimArtifact(1:numel(StimTrain));
                                if idx(1)+length(StimArtifact) < size(SimArtifact,1)
                                    SimArtifact(idx(1):(idx(1)+length(StimArtifact)-1),thisElectrode,thisArray) = SimArtifact(idx(1):(idx(1)+length(StimArtifact)-1),thisElectrode,thisArray) + StimArtifact';
                                else
                                    SimArtifact(idx(1):end,thisElectrode,thisArray) = SimArtifact(idx(1):end,thisElectrode,thisArray) + StimArtifact(1:(size(SimArtifact,1)-idx(1)+1))';
                                end
                            end
                        end
                        % end of hand crafted stimulation artifact simulation
                    case 2 % RC circuit simulation
                        % stimulation setup
                        % generate stimulation pulse train waveforms
                        % first generate stimulation pulse train per frame
                        frameStimTrain = zeros(FrameTime * 10^6,1);
                        StimPulsePeriod = 1/StimParam.Frequency * 10^6;
                        NumPulses = floor(FrameTime * 10^6 / StimPulsePeriod);

                        for thisPulse = 1:NumPulses
                            phase1idx = (1:(StimParam.timeperphase*1000)) + (thisPulse-1) * StimPulsePeriod;
                            phase2idx = (1:(StimParam.timeperphase*1000)) + (thisPulse-1) * StimPulsePeriod+StimParam.timeperphase*1000+StimParam.timeinterphase*1000;
                            frameStimTrain(phase1idx) = -1*(1+randn(1)*ArtifactNoiseLevel); % one way to simulate the observed noise level in the Blackrock recording
                            frameStimTrain(phase2idx) = 1*(1+randn(1)*ArtifactNoiseLevel);
                        end

                        % put trains together
                        StimTrain = [];
                        for thisFrame = 1:NumFrames
                            StimTrain = [StimTrain;frameStimTrain*StimParam.CurrentLevelFerFrame(thisFrame)];
                        end

                        % downsample data
                        nsample = floor(numel(StimTrain)/10^6*RawDataSampleRate);
                        idx = linspace(1,numel(StimTrain),nsample);idx = round(idx);
                        StimTrain = StimTrain(idx);

                        % simulate stimulation artifact shapes by simulating a simple RC circuit.
                        % The stimulation artifact is modeled as the capacitor V. Feel free to
                        % replace it with real recorded artifacts.

                        vm = 0;
                        R = (ElectrodeImpedance + TissueImpedanceperMM*Distance) * 10^3; % improve with real values from measurement
                        C = (ElectrodeCapacitance + TissueCapacitanceperMM*Distance) *10^-9; % improve with real values from measurement
                        tau = R*C; % Time constant
                        t_input_start = 0;
                        dt_input = 1/RawDataSampleRate; % Time step for input data
                        t_input_end = (numel(StimTrain)-1) * dt_input; % input data
                        t_input = (t_input_start : dt_input : t_input_end)'; % Column vector
                        V = @(t) interp1(t_input, StimTrain, t, 'linear', 'extrap');
                        V_C0 = 0; % Assuming capacitor is initially uncharged
                        ode_func = @(t, V_C) (1/(R*C)) * (V(t) - V_C);
                        tspan = [0:dt_input:SimulationLength];
                        [t_solution, V_C_solution] = ode45(ode_func, tspan, V_C0);
                        % Calculate Current I(t) = (V_in(t) - V_C(t)) / R
                        V_in_at_solution_times = V(t_solution);
                        I_solution = (V_in_at_solution_times - V_C_solution) / R;

                        StimArtifact = V_C_solution; % this will be the base shape of the artifact, it is supposed to be very similar on nearby electrodes but can be different on distant electrodes.
                        StimArtifact = StimArtifact(1:size(SimArtifact,1));
                        SimArtifact(:,thisElectrode,thisArray) = SimArtifact(:,thisElectrode,thisArray) + StimArtifact;
                        % in reality there is sampling aliasing but it is not simulated here, improve

                        % Plot Results
                        if plotArtifact
                            figure;

                            subplot(3,1,1);
                            plot(t_input, StimTrain, 'b', 'LineWidth', 1.5); % Plot the original input data
                            hold on;
                            plot(t_solution, V_in_at_solution_times, 'b:', 'LineWidth', 0.8); % Plot interpolated V_in
                            title('Input Voltage V_{in}(t)');
                            xlabel('Time (s)');
                            ylabel('Voltage (V)');
                            legend('Original Input Data', 'Input at Solution Times (Interpolated)');
                            grid on;

                            subplot(3,1,2);
                            plot(t_solution, V_C_solution, 'r', 'LineWidth', 1.5);
                            title('Capacitor Voltage V_C(t)');
                            xlabel('Time (s)');
                            ylabel('Voltage (V)');
                            grid on;

                            subplot(3,1,3);
                            plot(t_solution, I_solution, 'g', 'LineWidth', 1.5);
                            title('Circuit Current I(t)');
                            xlabel('Time (s)');
                            ylabel('Current (A)');
                            grid on;

                            sgtitle(sprintf('RC Circuit Response to Arbitrary Input (R=%.0f \\Omega, C=%.1g F, \\tau=%.2g s)', R, C, tau));
                        end

                    case 3 % real recorded artifact
                        % only used during reviewing but not included
                        % here
                end

            end
        end % end of stimulation artifact simulation
        initialized = true;
    end
end

% LFP and spike waveforms only need to generate once because in this simple
% simulation they only depends on spike timing

% simple LFP simulation, random pink noise + spike triggered LFP
fprintf('Simulating LFP...\n')
for thisArray = 1:NumRecordingArray
    for thisElectrode = 1:RecordingElecPerArray
        % first generate random pink noise on every channel
        m = 1;
        n = size(SimLFP,1);
        baselineLFP = ArrayParam(thisArray).LFPbaselineSigma(thisElectrode) * pinknoise(m,n);
        baselineLFP = baselineLFP';
        % calculate spike triggered LFP
        % for every electrode, there is a spike triggered LFP component on
        % the current electrode based on distance
        % first calculate the spatial factor
        SpikeTriggeredLFP0 = zeros(size(SimLFP,1),1);
        for thisArray2 = 1:NumRecordingArray
            for thisElectrode2 = 1:RecordingElecPerArray
                x1 = ArrayParam(thisArray).Xmmcoord(thisElectrode);
                y1 = ArrayParam(thisArray).Ymmcoord(thisElectrode);
                x2 = ArrayParam(thisArray).Xmmcoord(thisElectrode2);
                y2 = ArrayParam(thisArray).Ymmcoord(thisElectrode2);
                Distance = ((x1-x2)^2 + (y1-y2)^2)^0.5;

                % for the sitimulating electrode
                if Distance < 0.1
                    Distance = 0.1;
                end

                SpatialFactor = gaussmf(Distance,[0.3 0]);
                if SpatialFactor > 0.01
                    SpikeTrain = SimSpikeTrain(:,thisElectrode2,thisArray2);
                    % spike triggered LFP kernel with some random factors
                    x = -100:1/30:300; % ms
                    r1 = random('Normal',2,0.2);
                    y1 = -gaussmf(x,[r1 r1]);
                    r1 = random('Normal',60,6);
                    r2 = random('Normal',100,10);
                    y2 = gaussmf(x,[r1 r2]);
                    r1 = random('Normal',30,3);
                    r2 = random('Normal',20,2);
                    y3 = -gaussmf(x,[r1 r2]);
                    r1 = random('Normal',0.5,0.05);
                    r2 = random('Normal',0.3,0.03);
                    r3 = random('Normal',0.4,0.04);
                    SpikeTriggeredLFP = (0.5*y1+0.3*y2 + 0.4* y3)*5;
                    %                     figure
                    %                     plot(SpikeTriggeredLFP)
                    SpikeTriggeredLFP = conv(SpikeTrain,SpatialFactor*SpikeTriggeredLFP);
                    SpikeTriggeredLFP = SpikeTriggeredLFP(1:numel(SpikeTrain));
                    SpikeTriggeredLFP0 = SpikeTriggeredLFP0 + SpikeTriggeredLFP;
                end
            end
        end
        % combine spike triggered LFP to the base line LFP
        simLFP0 = baselineLFP + SpikeTriggeredLFP0;
        SimLFP(:,thisElectrode,thisArray) = simLFP0;
    end
end

% simple spike waveform simulation, can be improved
fprintf('Simulating spikew waveforms...\n')
for thisArray = 1:NumRecordingArray
    for thisElectrode = 1:RecordingElecPerArray
        tempdata = SimLFP(:,thisElectrode,thisArray); % used to calculate base noise level
        tempdata = filtfilt(Bhigh, Ahigh,tempdata);

        AllSNR(thisElectrode,thisArray) = SpikeWaveformSNR; % can be randomized
        % draw a random number form 1 to number of AllSpikeWaveform
        NumSpikeWaveform = size(AllSpikeWaveform,2);
        thisSpikeWavefrom = randi(NumSpikeWaveform);
        SpikeWaveform = AllSpikeWaveform(:,thisSpikeWavefrom);

        SpikeWaveform = SpikeWaveform * SpikeWaveformSNR * std(tempdata);


        % process spike train
        SpikeTrain = SimSpikeTrain(:,thisElectrode,thisArray);
        % assign random size to spikes
        SpikeIdx = find(SpikeTrain);
        for thisSpike = 1:numel(SpikeIdx)
            r1 = random('Uniform',1,1.5);
            SpikeTrain(SpikeIdx(thisSpike)) = r1;
        end
        % convolve with spike waveform
        SpikeWaveforms = conv(SpikeTrain,SpikeWaveform);
        SpikeWaveforms = SpikeWaveforms(1:numel(SpikeTrain));
        % broad band signal
        data = SpikeWaveforms + SimLFP(:,thisElectrode,thisArray);
        %data = filtfilt(Bhigh, Ahigh, data);
        SimBB(:,thisElectrode,thisArray) = data;
    end
end


SimCombined = SimBB + SimArtifact;
save('./SimulatedData_lower_freq.mat','AllStimIdx','SimFR','SimSpikeTrain','SimLFP','SimBB','SimArtifact','SimCombined','StimParam','AllSNR')

% SCRIPT TO PLOT **ONLY THE ARTIFACT**
clear;
clc;
data_file_path = '/Users/bui/code/sparc/research/generate_dataset/SimulatedData_lower_freq.mat';
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