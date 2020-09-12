% this code is used to create the folder of SAAM. Each video will be
% extracted into folder containing frames not avi
% 

INP_FOLDER = './output_SDI/SMIC_MAG_FRAMES';
OUT_FOLDER = 'SMIC_4CLASSES';
mkdir(OUT_FOLDER)

pos_OUT_FOLDER = fullfile(OUT_FOLDER , 'Positive');
mkdir(pos_OUT_FOLDER);
neg_OUT_FOLDER = fullfile(OUT_FOLDER , 'Negative');
mkdir(neg_OUT_FOLDER);
sur_OUT_FOLDER = fullfile(OUT_FOLDER , 'Surprise');
mkdir(sur_OUT_FOLDER);

list_subject = dir(INP_FOLDER)
n_subject = length(list_subject);

for i=3:n_subject
	subject_name = list_subject(i).name;
	subject_path = fullfile(INP_FOLDER,subject_name);
	
	list_video = dir(subject_path);
	
	n_video = length(list_video);
	
	for j=3:n_video
		video_name = list_video(j).name;
		video_path = fullfile(subject_path , video_name)
		
		list_img = dir(video_path);
		
		img_path = fullfile(video_path , list_img(3).name );
		img_name  = list_img(3);
		img = imread(img_path);
		
		emotion = split(video_name,'_');
		str_emo = emotion{2};
		
		if (str_emo(1) == 's')
			out_img_path = ['Sur' , '-' , video_name , '.jpg'];
			out_img_path = fullfile(sur_OUT_FOLDER , out_img_path);
		else
			if (str_emo(1) == 'p')
				out_img_path = ['Pos' , '-' , video_name , '.jpg'];
				out_img_path = fullfile(pos_OUT_FOLDER , out_img_path);
			else
				out_img_path = ['Neg' , '-' , video_name , '.jpg'];
				out_img_path = fullfile(neg_OUT_FOLDER , out_img_path);
			end
		
		end
		
		imwrite(img, out_img_path);

	end
	
end