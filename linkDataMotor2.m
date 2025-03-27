function linkDataMotor2(patNow)

% EXAMPLE: patNow = 'S23_10b';
rawFolder = ['/home/klab/NAS/SEEG-RAW-DATA/Video/', patNow, '/'];
outputSubjDir = '/home/klab/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/videoDateTimes/Raw CSVs/';

% Set processed folders
videoFilenamesTrack = [];
videoStartTimeTrack = [];
videoEndTimeTrack = [];
eegFiles  =  dir([rawFolder, '/*.EEG']);
cnt = 1;

for s = 1:length(eegFiles)
    timeFile = fullfile(rawFolder, [eegFiles(s).name(1:end-4), '.CN3']);
    fileID = fopen(timeFile,'r'); 
    txtInfo = fscanf(fileID, '%s');
    dateFound = num2str(str2num(txtInfo(60:80)));
    datetimeFound = datetime(dateFound,'InputFormat','yyyyMMddHHmmssS');
    videoFiles = dir([rawFolder, '/', eegFiles(s).name(1:end-4), '.VOR', '/*.m2t']);
    videoStarTime = []; videoEndTime = [];

    for i = 1:length(videoFiles)
        videoFile = fullfile(videoFiles(i).folder, videoFiles(i).name);
        disp(videoFile);
        vidObj=VideoReader(videoFile);
        
        if i == 1
            videoStarTime = datetimeFound;
            videoEndTime = datetimeFound + seconds(vidObj.Duration);

        elseif i == 2
            videoStarTime = videoEndTime;
            videoEndTime = videoStarTime + seconds(vidObj.Duration);
        elseif i ==3
            disp('this are 3 videos folders');
        end
        % EXTRACT VIDEO TIMING
        videoFilenamesTrack{cnt} = videoFiles(i).name;
        videoStartTimeTrack{cnt} = videoStarTime;
        videoEndTimeTrack{cnt} = videoEndTime;
        
        cnt = cnt + 1;
    end
end

videoFileTable = table(videoFilenamesTrack',videoStartTimeTrack',videoEndTimeTrack', ...
    'VariableNames',["Filename","VideoStart","VideoEnd"]);
fileTableName = ['videoFileTable_', extractBetween(patNow, 1, 1), extractBetween(patNow, 5, length(patNow)), '.csv'];
fileTableName = strjoin(fileTableName, '');
fullFileName = fullfile(outputSubjDir,fileTableName);
disp(fullFileName);
disp(videoFileTable);
writetable(videoFileTable,fullFileName);