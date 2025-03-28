function linkDataMotor2(patNow)

% EXAMPLE: patNow = 'S23_10b';
videoFolder = ['/home/klab/NAS/SEEG-RAW-DATA/Video/', patNow, '/'];
timingFolder = ['/home/klab/NAS/SEEG-RAW-DATA/Timing/', patNow, '/'];
outputSubjDir = '/home/klab/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/videoDateTimes/Raw CSVs/';

% Initialize containers
videoFilenamesTrack = {};
videoStartTimeTrack = {};
videoEndTimeTrack = {};

% Get first available .CN3 file
cn3Files = dir(fullfile(timingFolder, '*.CN3'));
if isempty(cn3Files)
    error('No CN3 timing file found in %s', timingFolder);
end

timeFile = fullfile(timingFolder, cn3Files(1).name);
fileID = fopen(timeFile, 'r'); 
txtInfo = fscanf(fileID, '%s');
fclose(fileID);

% Parse timestamp
dateFound = num2str(str2num(txtInfo(60:80)));
datetimeFound = datetime(dateFound, 'InputFormat', 'yyyyMMddHHmmssS');
videoStartTime = datetimeFound;

% Process .m2t video files
videoFiles = dir(fullfile(videoFolder, '*.m2t'));
disp(['Found video files: ', num2str(length(videoFiles))]);

for i = 1:length(videoFiles)
    videoFile = fullfile(videoFiles(i).folder, videoFiles(i).name);
    disp(['Processing video: ', videoFile]);

    vidObj = VideoReader(videoFile);
    videoFilenamesTrack{i} = videoFiles(i).name;
    videoStartTimeTrack{i} = videoStartTime;
    videoEndTime = videoStartTime + seconds(vidObj.Duration);
    videoEndTimeTrack{i} = videoEndTime;

    videoStartTime = videoEndTime;  % Prepare for next video
end

% Create and save table
videoFileTable = table(videoFilenamesTrack', videoStartTimeTrack', videoEndTimeTrack', ...
    'VariableNames', ["Filename", "VideoStart", "VideoEnd"]);

fileTableName = ['videoFileTable_', extractBetween(patNow, 1, 1), extractBetween(patNow, 5, length(patNow)), '.csv'];
fileTableName = strjoin(fileTableName, '');
fullFileName = fullfile(outputSubjDir, fileTableName);

disp(fullFileName);
disp(videoFileTable);
writetable(videoFileTable, fullFileName);