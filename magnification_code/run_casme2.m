
% TRANG,PLESAE SET UPTWO FOLDER HERE
SRC_FOLDER = 'Cropped';
DST_FOLDER = 'OUT_MAG_20';
mkdir(DST_FOLDER)

addpath('EVM_Matlab');
addpath('EVM_Matlab//matlabPyrTools');

list_subject = dir(SRC_FOLDER)
n_subject = length(list_subject);

for i = 3: n_subject
    subject_path = fullfile(SRC_FOLDER , list_subject(i).name);
    dst_subject_path = fullfile(DST_FOLDER, list_subject(i).name);
    
    % create subject path 
    mkdir(dst_subject_path);
    
    list_video = dir(subject_path);
    
    % process videos of subject
    for j = 3: length(list_video)
        video_path = fullfile(subject_path ,list_video(j).name );
        dst_video_path = fullfile(dst_subject_path , list_video(j).name);
        
        mkdir(dst_video_path);
        
        imageNames = dir(fullfile(video_path,'*.jpg'));
        imageNames = {imageNames.name}';
        
        outputVideo = VideoWriter(fullfile(video_path,'video_out.avi'));
        outputVideo.FrameRate = 200;

        open(outputVideo)

        for ii = 1:length(imageNames)
           img = imread(fullfile(video_path,imageNames{ii}));
           writeVideo(outputVideo,img)
        end

        close(outputVideo);
        
        % strat to use motion mag
        inFile = fullfile(video_path , 'video_out.avi');
        amplify_spatial_lpyr_temporal_iir(inFile, dst_video_path , 20 , 16, 0.4, 0.05, 0.1);
    end
    
end

for i = 3: n_subject
    
    dst_subject_path = fullfile(DST_FOLDER, list_subject(i).name);
    
    % create subject path 
    
    list_video = dir(subject_path);
    
    % process videos of subject
    for j = 3: length(list_video)
        
        dst_video_path = fullfile(dst_subject_path , list_video(j).name);
        
        
        imageNames = dir(fullfile(video_path,'*.jpg'));
        imageNames = {imageNames.name}';
        
        outputVideo = VideoWriter(fullfile(video_path,'video_out.avi'));
        outputVideo.FrameRate = 200;

        open(outputVideo)

        for ii = 1:length(imageNames)
           img = imread(fullfile(video_path,imageNames{ii}));
           writeVideo(outputVideo,img)
        end

        close(outputVideo);
        
        % strat to use motion mag
        inFile = fullfile(video_path , 'video_out.avi');
        amplify_spatial_lpyr_temporal_iir(inFile, dst_video_path , 20 , 16, 0.4, 0.05, 0.1);
    end
    
end

