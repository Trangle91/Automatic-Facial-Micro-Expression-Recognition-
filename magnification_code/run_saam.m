
% TRANG,PLESAE SET UPTWO FOLDER HERE
SRC_FOLDER = '/home/thtran/Project/ME/MER/magnification_code/SAMM_VIDEO/SAMM_VIDEO_BEFORE_MAG';
DST_FOLDER = 'OUT_MAG_SAAM';
mkdir(DST_FOLDER)

addpath('EVM_Matlab');
addpath('EVM_Matlab//matlabPyrTools');

list_subject = dir(SRC_FOLDER)
n_subject = length(list_subject);

for i = 14: n_subject
    subject_path = fullfile(SRC_FOLDER , list_subject(i).name);
    dst_subject_path = fullfile(DST_FOLDER, list_subject(i).name);
    
    % create subject path 
    mkdir(dst_subject_path);
    
    list_video = dir(subject_path);
    
    % process videos of subject
    for j = 3: length(list_video)
        video_path = fullfile(subject_path ,list_video(j).name );
		
		video_name = list_video(j).name;
		video_name_parts = split(video_name,'.');
        dst_video_path = fullfile(dst_subject_path , video_name_parts(1) )
        
        mkdir(dst_video_path{1});
        
        
        % strat to use motion mag

        amplify_spatial_lpyr_temporal_iir(video_path, dst_video_path{1} , 20 , 16, 0.4, 0.05, 0.1);
    end
    
end
