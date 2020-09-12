% create video based on images
workingDir = './test_data'
imageNames = dir(fullfile(workingDir,'*.jpg'));
imageNames = {imageNames.name}';

outputVideo = VideoWriter(fullfile(workingDir,'shuttle_out.avi'));
outputVideo.FrameRate = 200;
open(outputVideo)

for ii = 1:length(imageNames)
   img = imread(fullfile(workingDir,imageNames{ii}));
   writeVideo(outputVideo,img)
end

close(outputVideo);

shuttleAvi = VideoReader(fullfile(workingDir,'shuttle_out.avi'));
