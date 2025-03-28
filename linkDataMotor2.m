function linkDataMotor2(patNow)

% EXAMPLE: patNow = 'S23_10b';
videoFolder = ['/home/klab/NAS/SEEG-RAW-DATA/Video/', patNow, '/'];
timingFolder = ['/home/klab/NAS/SEEG-RAW-DATA/Timing/', patNow, '/'];
eegFolder = ['/home/klab/NAS/SEEG-RAW-DATA/iEEG/', patNow, '/'];
outputSubjDir = '/home/klab/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/videoDateTimes/Raw CSVs/';

% Set processed folders
videoFilenamesTrack = {};
videoStartTimeTrack = {};
videoEndTimeTrack = {};
cnt = 1;

eegFiles = dir([eegFolder, '/*.EEG']);

for s = 1:length(eegFiles)
    baseName = eegFiles(s).name(1:end-4);  % Remove .EEG
    timeFile = fullfile(timingFolder, [baseName, '.CN3']);

    if ~isfile(timeFile)
        warning('Missing timing file: %s', timeFile);
        continue;
    end

    fileID = fopen(timeFile,'r'); 
    txtInfo = fscanf(fileID, '%s');
    fclose(fileID);
    dateFound = num2str(str2num(txtInfo(60:80)));
    datetimeFound = datetime(dateFound,'InputFormat','yyyyMMddHHmmssS');

    videoFiles = dir(fullfile(videoFolder, '*.m2t'));
    if isempty(videoFiles)
        warning('No .m2t files found in %s', videoFolder);
        continue;
    end

    videoStartTime = datetimeFound;

    for i = 1:length(videoFiles)
        videoFile = fullfile(videoFiles(i).folder, videoFiles(i).name);
        disp(videoFile);
        vidObj = VideoReader(videoFile);

        videoFilenamesTrack{cnt} = videoFiles(i).name;
        videoStartTimeTrack{cnt} = videoStartTime;
        videoEndTime = videoStartTime + seconds(vidObj.Duration);
        videoEndTimeTrack{cnt} = videoEndTime;

        videoStartTime = videoEndTime;  % Prepare for next file
        cnt = cnt + 1;
    end
end

videoFileTable = table(videoFilenamesTrack', videoStartTimeTrack', videoEndTimeTrack', ...
    'VariableNames', ["Filename", "VideoStart", "VideoEnd"]);
fileTableName = ['videoFileTable_', extractBetween(patNow, 1, 1), extractBetween(patNow, 5, length(patNow)), '.csv'];
fileTableName = strjoin(fileTableName, '');
fullFileName = fullfile(outputSubjDir, fileTableName);

disp(fullFileName);
disp(videoFileTable);
writetable(videoFileTable, fullFileName);