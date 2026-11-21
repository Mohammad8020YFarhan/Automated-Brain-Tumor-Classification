clear; clc; close all;

%%
datasetRoot='C:\Users\ArmanComputer\Desktop\Article\تومور مغزی\brain';
outputRoot    = 'E:\';
targetSize    = [224 224];
classNames    = {'glioma','meningioma','notumor','pituitary'};
subsets       = {'Training','Testing'};

%%
if ~exist(outputRoot, 'dir')
    mkdir(outputRoot);
end
for s = 1:numel(subsets)
    for c = 1:numel(classNames)
        outDir = fullfile(outputRoot, subsets{s}, classNames{c});
        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end
    end
end

%%
totalProcessed = 0;
totalFailed    = 0;

for s = 1:numel(subsets)
    subsetName = subsets{s};
    fprintf('\n=== در حال پردازش مجموعه: %s ===\n', subsetName);

    for c = 1:numel(classNames)
        className = classNames{c};
        inDir  = fullfile(datasetRoot, subsetName, className);
        outDir = fullfile(outputRoot,  subsetName, className);

        if ~exist(inDir, 'dir')
            warning('پوشه یافت نشد: %s', inDir);
            continue;
        end

        imgFiles = [dir(fullfile(inDir,'*.jpg')); dir(fullfile(inDir,'*.jpeg')); ...
                    dir(fullfile(inDir,'*.png'));  dir(fullfile(inDir,'*.bmp'))];

        fprintf('  کلاس %-12s : %d تصویر\n', className, numel(imgFiles));

        for k = 1:numel(imgFiles)
            try
                srcPath = fullfile(imgFiles(k).folder, imgFiles(k).name);
                img = imread(srcPath);

                processedImg = preprocessSingleImage(img, targetSize);

                [~, baseName, ~] = fileparts(imgFiles(k).name);
                dstPath = fullfile(outDir, [baseName '.png']);
                imwrite(processedImg, dstPath);

                totalProcessed = totalProcessed + 1;
            catch ME
                fprintf('    خطا در فایل %s: %s\n', imgFiles(k).name, ME.message);
                totalFailed = totalFailed + 1;
            end
        end
    end
end

fprintf('\n=== پایان پیش‌پردازش ===\n');
fprintf('تعداد موفق : %d\n', totalProcessed);
fprintf('تعداد ناموفق: %d\n', totalFailed);
fprintf('خروجی در: %s\n', outputRoot);


%% ========================================================================
% ========================================================================
function outImg = preprocessSingleImage(img, targetSize)

    if size(img,3) == 1
        grayImg = img;
    elseif size(img,3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img(:,:,1);
    end

    cropBox = findBrainBoundingBox(grayImg);
    grayCropped = imcrop(grayImg, cropBox);

    if isempty(grayCropped) || numel(grayCropped) < 100
        grayCropped = grayImg;
    end

    denoised = medfilt2(grayCropped, [3 3]);

    enhanced = adapthisteq(denoised, 'ClipLimit', 0.01, 'Distribution', 'rayleigh');

    resized = imresize(enhanced, targetSize);

    outImg = cat(3, resized, resized, resized);
end


%% ========================================================================
% Otsu thresholding
% ========================================================================
function bbox = findBrainBoundingBox(grayImg)

    level = graythresh(grayImg);
    bw = imbinarize(grayImg, level);

    bw = bwareaopen(bw, 50);
    bw = imfill(bw, 'holes');

    cc = bwconncomp(bw);
    if cc.NumObjects == 0
        bbox = [1, 1, size(grayImg,2)-1, size(grayImg,1)-1];
        return;
    end

    stats = regionprops(cc, 'Area', 'BoundingBox');
    [~, idx] = max([stats.Area]);
    bbox = stats(idx).BoundingBox;

    padding = 5;
    bbox(1) = max(1, bbox(1) - padding);
    bbox(2) = max(1, bbox(2) - padding);
    bbox(3) = bbox(3) + 2*padding;
    bbox(4) = bbox(4) + 2*padding;
end