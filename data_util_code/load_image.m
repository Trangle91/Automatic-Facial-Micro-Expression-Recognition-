%% Function for loading images in a directory
function [imv,imSize]=load_image(imDir)

skipInDirList = 3;  %skip . and .. in directory lists

% Inspect first image and prealloate for speed
imDirList = dir(imDir);
imDirListLength = length(imDirList);

imFile = imDirList(skipInDirList+1).name
im = imread(strcat(imDir, '/', imFile));
imSize = size(im);

imv = zeros(size(im(:), 1), imDirListLength-skipInDirList);

% Load images
for i = (skipInDirList+1):imDirListLength
    imFile = imDirList(i).name;
    imPath = strcat(imDir, '/', imFile)
    im = imread(imPath);
    imv(:, i-skipInDirList) = im(:);  %vectorise image
end