% this code is used to create the folder of SAAM. Each video will be
% extracted into folder containing frames not avi
% 
FOLDER_MAG_SMIC = './magnification_code/OUT_MAG_SMIC/';
OUT_FOLDER = './OUTFOLD_MAG_SMIC';
mkdir(OUT_FOLDER)

list_subject = dir(FOLDER_MAG_SMIC);
n_subject = length(list_subject)

for i=1:n_subject
    curr_name = list_subject(i).name;
    if (curr_name(1) == '.')
        continue;
    end
    
    in_folder_subject = fullfile(FOLDER_MAG_SMIC , curr_name);
    ou_folder_subject = fullfile(OUT_FOLDER , curr_name);
    mkdir(ou_folder_subject);
    
    list_video = dir(in_folder_subject);
    n_video = length(list_video);
    for j=3:n_video
        curr_video = list_video(j).name;
        if (curr_video(1) == '.')
            continue;
        end
        in_folder_video = fullfile(in_folder_subject ,curr_video )
        
        ou_folder_video = fullfile(ou_folder_subject , curr_video )
        
        mkdir(ou_folder_video);
        
        list_file = dir(in_folder_video);
        n_file = length(list_file);
        for k=3:n_file
           file_name = list_file(k).name;
           fullpath_filename = fullfile(in_folder_video, file_name)
           vobj = VideoReader(fullpath_filename);
           idx = 1;
           while hasFrame(vobj)
                frame = readFrame(vobj);
                
                if (idx < 10)
                   img_name = ['img00' , num2str(idx) , '.jpg']; 
                else
                    if (idx < 100)
                        img_name = ['img0' , num2str(idx) , '.jpg']; 
                    else
                        img_name = ['img' , num2str(idx) , '.jpg']; 
                    end
                end
                full_path_img = fullfile(ou_folder_video ,img_name )
                imwrite(frame, full_path_img);
                idx = idx + 1;
           end
        end
        
    end
    
    
end





