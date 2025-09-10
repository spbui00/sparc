clear
clear variables; 
clc

% hyperparameters
SVD_threshold = 0.9;

% data load and rescale to [-1,1]
addpath('../stanford_dataset/eraasr-1.0.0/')
dat=load('exampleDataTensor.mat');
data_in=dat.data_trials_by_time_by_channels;
% Ain = -1 + 2.*(data_in - min(data_in))./(max(data_in) - min(data_in));
% data_in = permute(data_in, [3 2 1]);


% Ain = permute(Ain, [3 2 1]);
figure
plot(squeeze(data_in(1,:,:)))

%% dictionary learning parameters
dataInt = data_in; %time x channels x epochs

fsData = 30000; % sampling rate of the data Hz

stimChans = [];% the channels used for stimulation . These should be noted and excluded from further analysis

plotIt = 0;% determines whether or not to plot the intermediate results of the functions.

tEpoch = 0.25;% epoched time window (s)


type = 'dictionary';
useFixedEnd = 0; % if wanting to use a fixed distance, rather than dynamically detecting them

% parameters for detecting the onset and offset of artifacts
pre = 50; % default time window to extend before the artifact pulse to ensure the artifact is appropriately detected (0.8 ms as default)
post = 200; % default time window to extend before the artifact pulse to ensure the artifact is appropriately detected (1 ms as default)
fixedDistance = 4; % in ms, duration to extract to detect the artifact pulse. Setting this too short may result in not detecting the artifact
recoverExp = 0; % optional parameters for exponential recovery, not currently used. Could be helpful for signals with large exponential recoveries
expThreshVoltageCut = 95;
expThreshDiffCut = 95;
threshVoltageCut = 75;
threshDiffCut = 75;
onsetThreshold = 1.5;
chanInt = 1;
chanIntList = [1]; % these are the channels of interest to visualize in closer detail

% additional HDBSCAN parameters and window selection
minDuration = 0.5; % minimum duration of artifact in ms
bracketRange = [-6:6]; %This variable sets the number of samples around the maximum voltage deflection to use for template clustering and subsequent matching. The smaller this range, the lower the dimensionality used for clustering, and the fewer points used to calculate the best matching template. This value is used to try and ensure that non-informative points are not included in the clustering and template matching. This should be set to what looks like the approximate length of the artifact's largest part.
minPts = 2;
minClustSize = 3;
outlierThresh = 0.95;

% these are the metrics used if the dictionary method is selected. The
% options are 'eucl', 'cosine', 'corr', for either euclidean distance,
% cosine similarity, or correlation for clustering and template matching.
distanceMetricDbscan = 'eucl';
distanceMetricSigMatch = 'corr';
amntPreAverage = 3;
normalize = 'preAverage';

%% %% dictionary learning and visualization
[processedSig,templateDictCell,templateTrial,startInds,endInds] = analyFunc.template_subtract(dataInt,'type',type,...
        'fs',fsData,'plotIt',plotIt,'pre',pre,'post',post,'stimChans',stimChans,...
        'useFixedEnd',useFixedEnd,'fixedDistance',fixedDistance,...,
        'distanceMetricDbscan',distanceMetricDbscan,'distanceMetricSigMatch',distanceMetricSigMatch,...
        'recoverExp',recoverExp,'normalize',normalize,'amntPreAverage',amntPreAverage,...
        'minDuration',minDuration,'bracketRange',bracketRange,'threshVoltageCut',threshVoltageCut,...
        'threshDiffCut',threshDiffCut,'expThreshVoltageCut',expThreshVoltageCut,...
        'expThreshDiffCut',expThreshDiffCut,'onsetThreshold',onsetThreshold,'chanInt',chanInt,...
        'minPts',minPts,'minClustSize',minClustSize,'outlierThresh',outlierThresh);

% visualization
% of note - more visualizations are created here, including what the
% templates look like on each channel, and what the discovered templates are
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
average = 1;
modePlot = 'avg';
xlims = [-200 1000];
ylims = [-0.6 0.6];
vizFunc.small_multiples_time_series(processedSig,tEpoch,'type1',stimChans,'type2',0,'xlims',xlims,'ylims',ylims,'modePlot',modePlot,'highlightRange',trainDuration)