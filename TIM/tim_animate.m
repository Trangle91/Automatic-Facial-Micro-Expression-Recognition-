%% Normalising an image sequence through TIM.
%%
%% Copyright Tomas Pfister 2011
%% Conditions for use: see license.txt
%%
%% Usage: tim_animate(srcdir, dstdir, length_of_normalised_sequence)

function tim_animate(imDir, outputDir, expandNum)

% Load images into a vector
[imv,imSize] = loadImages(imDir);

% Create animation
aniModel = tim_getAniModel(imv);
aniData = tim_genAnimationData(aniModel, 0, 1, expandNum);

% Extract interpolated image sequence
for i = 1:expandNum
    img = reshape(aniData(:, i), imSize);
    img = uint8(img);
    imwrite(img, strcat(outputDir, '/', num2str(i), '.bmp'));
end


%% Function for loading images in a directory
function [imv,imSize]=loadImages(imDir)

skipInDirList = 2;  %skip . and .. in directory lists

% Inspect first image and prealloate for speed
imDirList = dir(imDir);
imDirListLength = length(imDirList);

imFile = imDirList(skipInDirList+1).name;
im = imread(strcat(imDir, '/', imFile));
imSize = size(im);

imv = zeros(size(im(:), 1), imDirListLength-skipInDirList);

% Load images
for i = (skipInDirList+1):imDirListLength
    imFile = imDirList(i).name;
    imPath = strcat(imDir, '/', imFile);
    im = imread(imPath);
    imv(:, i-skipInDirList) = im(:);  %vectorise image
end
