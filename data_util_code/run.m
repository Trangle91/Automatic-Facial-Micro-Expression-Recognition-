addpath TIM;
addpath TIM/TIM;

% setup the value and input Directory
expandNum = 20;
input_dir = 'EP01_5';

% create the output directory 
output_dir = [num2str(expandNum) , '_' , input_dir ];
mkdir(output_dir);

% Load images into a vector
[imv,imSize] = load_image(input_dir);


% Create animation
aniModel = tim_getAniModel(imv);
aniData = tim_genAnimationData(aniModel, 0, 1, expandNum);

% Extract interpolated image sequence
for i = 1:expandNum
    img = reshape(aniData(:, i), imSize);
    img = uint8(img);
    imwrite(img, strcat(output_dir, '/', num2str(i), '.jpg'));
end



