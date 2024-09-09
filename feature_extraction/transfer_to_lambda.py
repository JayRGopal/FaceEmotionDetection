/mnt/NAS/SEEG-RAW-DATA/S24_231/EEG2100/EA63018F.VOR/63018F00.m2t
Error using VideoReader/initReader (line 734)
Could not read file due to an unexpected error. Reason: Unable to initialize the video properties

Error in audiovideo.internal.IVideoReader (line 136)
            initReader(obj, fileName, currentTime);

Error in VideoReader (line 104)
            obj@audiovideo.internal.IVideoReader(varargin{:});

Error in linkDataMotor (line 26)
        vidObj=VideoReader(videoFile);