% This script is modified to generate ONLY the stimulation artifact signal.
% It is configured for a 4-second duration at a 512 Hz sampling rate.
clc, clear, close all

% --- Core Parameters (Set for your iEEG data) ---
SimulationLength = 4;       % The total simulated data length in seconds.
RawDataSampleRate = 30000;    % Your iEEG sampling rate.
NumRecordingArray = 2;      % Use 2 arrays to get enough channels.
RecordingElecPerArray = 64; % Each array has 64 electrodes.
ArtifactSimulationMethod = 1; % Use the hand-crafted method.
% --- End Core Parameters ---

% --- Other Simulation Parameters ---
ElectrodeImpedance = 150;
TissueImpedanceperMM = 6;
TissueCapacitanceperMM = 1.5;
ElectrodeCapacitance = 1;
FrameRate = 25;
FrameTime = 1/FrameRate;
NumFrames = SimulationLength/FrameTime;
PhospheneThreshold = 40;
MaxCurrentLevelMultiplier = 2.5;
GreyLevels = 8;
StimParam.Frequency = 200;
StimParam.Period = 1/StimParam.Frequency;
StimParam.timeperphase = 0.17;
StimParam.timeinterphase = 0.06;
StimParam.CurrentLevelFerFrame = randi(GreyLevels,NumFrames,1) / GreyLevels * PhospheneThreshold * MaxCurrentLevelMultiplier;
ArtifactNoiseLevel = 0.01;
SampleTime = 1/RawDataSampleRate;
StimulatedArray = 1:16:16;
StimulatingElectrode = 1;
plotArtifact = true;
% --- End Other Parameters ---

% Load electrode geometry
load('./ArrayParam.mat')

% Build frame stimulation timing info
LastTimeStampIdx = 0;
AllStimIdx = [];
for thisFrame = 1:NumFrames
    FrameStartTime = (thisFrame-1) * FrameTime;
    timeStamps = 0:SampleTime:(FrameTime-SampleTime);
    idx = 1:numel(timeStamps);
    idx = idx + LastTimeStampIdx;
    StimTime = 0:StimParam.Period:(FrameTime-StimParam.Period);
    StimIdx = round(StimTime/SampleTime)+1;
    StimIdx = StimIdx + LastTimeStampIdx;
    LastTimeStampIdx = idx(end);
    AllStimIdx = [AllStimIdx, StimIdx];
end

% Build electrode data tensor - ONLY for the artifact
total_samples = ceil(NumFrames * FrameTime * RawDataSampleRate);
SimArtifact = zeros(total_samples, RecordingElecPerArray, NumRecordingArray);

initialized = false;
for StimArray = StimulatedArray
    for StimElec = StimulatingElectrode
        disp(['Simulating Artifacts for: Array=',num2str(StimArray),' Elec=',num2str(StimElec)])
        
        fprintf('Simulating artifacts...\n')
        for thisArray = 1:NumRecordingArray
            for thisElectrode = 1:RecordingElecPerArray
                switch ArtifactSimulationMethod
                    case 1 % hand crafted artifact
                        withinframesizevar = 0.2;
                        amplitudeshiftingintime = 0.12;
                        beta0 = [0.62, 1.22];
                        % artifact spatial profile
                        x1 = ArrayParam(thisArray).Xmmcoord(thisElectrode);
                        y1 = ArrayParam(thisArray).Ymmcoord(thisElectrode);
                        x2 = ArrayParam(StimArray).Xmmcoord(StimElec);
                        y2 = ArrayParam(StimArray).Ymmcoord(StimElec);
                        Distance = ((x1-x2)^2 + (y1-y2)^2)^0.5;
                        if Distance < 0.1
                            Distance = 0.1;
                        end
                        LastTimeStampIdx = 0;
                        for thisFrame = 1:NumFrames
                            currentLevel = StimParam.CurrentLevelFerFrame(thisFrame);
                            yy = @(b,x) b(1) .*currentLevel.* (b(2).*x).^(-2);
                            yy2 = @(b,x) b(1) .*10.* (b(2).*x).^(-2);
                            r1 = random('Normal',beta0(1),beta0(1)/100);
                            r2 = random('Normal',beta0(2),beta0(2)/100);
                            beta(1) = r1;
                            beta(2) = r2;
                            SpatialArtifactFactor = yy(beta,Distance);
                            idx = 1:(FrameTime/SampleTime);
                            idx = idx + LastTimeStampIdx;
                            LastTimeStampIdx = idx(end);
                            if SpatialArtifactFactor > 0.001
                                n = round(random('Uniform',4, 6));
                                x = 0:(1/30):15; % ms
                                y1 = -sin(2*pi/x((n-1)*4)*(x(1:((n-1)*4))));
                                r3 = abs(random('Normal',0.4,0.04));
                                lamda1 = 1.5 - 1.5/2 * ((currentLevel/50)/(210/50));
                                lamda2 = 3 - 3/2 * ((currentLevel/50)/(210/50));
                                y2 = (exp(-lamda1*(x-x(n+1)))+exp(-lamda2*(x-x(n+1))));
                                y2 = y2/max(y2);
                                y = (y1(n)/abs(y1(n)))*ones(size(y2));
                                y(1:numel(y1)) = y1;
                                y((n+1):end) = y((n+1):end).*y2(n+1:end);
                                Kart = y;
                                Kart0 = Kart * SpatialArtifactFactor/yy2(beta,0.3);
                                StimTime = (0:StimParam.Period:(FrameTime-StimParam.Period))*1000;
                                StimIdx = round(StimTime/(1/30))+2;
                                StimTrain = zeros(1,round(FrameTime/SampleTime));
                                for thisStimPulse = 1:numel(StimIdx)
                                    if StimIdx(thisStimPulse) <= numel(StimTrain)
                                        StimTrain(StimIdx(thisStimPulse)) = random('Uniform',1-withinframesizevar/2,1+withinframesizevar/2);
                                    end
                                end
                                time0 = (thisFrame-1) * FrameTime;
                                time1 = thisFrame * FrameTime;
                                timeTicks = linspace(time0,time1,size(StimTrain,2));
                                signnum = rand(1)-0.5;
                                if signnum == 0; signnum = 1; else; signnum = signnum/abs(signnum); end
                                AmpTimeShift = signnum.*amplitudeshiftingintime.*sin(2.*pi.*(timeTicks/(rand(1)*(1)+0.1)) + rand(1).*2.*pi);
                                AmpTimeShift = AmpTimeShift + (1-AmpTimeShift(1));
                                StimTrain = StimTrain.*AmpTimeShift;
                                StimArtifact = conv(StimTrain,Kart0);
                                StimArtifact = StimArtifact(1:numel(StimTrain));
                                if idx(1)+length(StimArtifact) <= size(SimArtifact,1)+1
                                    SimArtifact(idx(1):(idx(1)+length(StimArtifact)-1),thisElectrode,thisArray) = SimArtifact(idx(1):(idx(1)+length(StimArtifact)-1),thisElectrode,thisArray) + StimArtifact';
                                else
                                    SimArtifact(idx(1):end,thisElectrode,thisArray) = SimArtifact(idx(1):end,thisElectrode,thisArray) + StimArtifact(1:(size(SimArtifact,1)-idx(1)+1))';
                                end
                            end
                        end
                        % --- END OF PASTED CODE ---
                    case 2 % RC circuit simulation
                        % This part is not used but kept for completeness
                end
            end
        end % end of stimulation artifact simulation
        initialized = true;
    end
end

% Save only the artifact signal
disp('Saving artifact-only data...');
save('./SimulatedArtifact.mat','SimArtifact','AllStimIdx','StimParam', '-v7.3')
disp('Done!');

% --- Plot for verification ---
figure;
time_vec = (0:size(SimArtifact,1)-1) / RawDataSampleRate;
plot(time_vec, SimArtifact(:, StimulatingElectrode, StimulatedArray));
title(['Final Artifact Signal (', num2str(RawDataSampleRate), ' Hz)']);
xlabel('Time (s)');
ylabel('Amplitude (uV)');
xlim([0 0.2]); % Zoom in to see details
grid on;