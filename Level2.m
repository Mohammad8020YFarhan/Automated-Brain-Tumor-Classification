clear; clc; close all;
rng(42);

%%
dataPath  = 'C:\Users\ArmanComputer\Desktop\Article\تومور مغزی';
modelFile = 'BrainTumor_ResNet50_DenseNet201_Ensemble.mat';
modelPath = fullfile(dataPath, modelFile);

if exist(modelPath, 'file') ~= 2
    error('فایل مدل ذخیره‌شده پیدا نشد:\n%s', modelPath);
end

fprintf('--> در حال بارگذاری مدل‌های ResNet50 و DenseNet201...\n');
loadedData = load(modelPath);

if ~isfield(loadedData, 'trainedNetResNet') || ~isfield(loadedData, 'trainedNetDenseNet')
    error('فایل MAT باید حاوی متغیرهای trainedNetResNet و trainedNetDenseNet باشد.');
end

netResNet   = loadedData.trainedNetResNet;
netDenseNet = loadedData.trainedNetDenseNet;
fprintf('مدل‌ها با موفقیت بارگذاری شدند.\n');

%%
trainDir = fullfile(dataPath, 'Training');
testDir  = fullfile(dataPath, 'Testing');

hasTrainFolder = (exist(trainDir, 'dir') == 7);
hasTestFolder  = (exist(testDir,  'dir') == 7);

if hasTrainFolder && hasTestFolder
    fprintf('--> ساختار مستقل Training و Testing شناسایی شد.\n');
    imdsTrainAll = imageDatastore(trainDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    imdsTest     = imageDatastore(testDir,  'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    
    [imdsTrain, ~] = splitEachLabel(imdsTrainAll, 0.85, 0.15, 'randomized');
    allClassNames  = categories(imdsTrainAll.Labels);
else
    fprintf('--> پوشه‌های مجزای Training/Testing یافت نشدند؛ بارگذاری مستقیم از پوشه‌های کلاس...\n');
    imdsAll = imageDatastore(dataPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    
    allClassNames = categories(imdsAll.Labels);
    if numel(allClassNames) < 2
        error('حداقل دو کلاس در مسیر مشخص‌شده یافت نشد. مسیر داده را بررسی کنید.');
    end
    
    [imdsTrain, ~, imdsTest] = splitEachLabel(imdsAll, 0.70, 0.15, 0.15, 'randomized');
end

classNames  = allClassNames;
trainLabels = normalizeLabels(imdsTrain.Labels, classNames);
testLabels  = normalizeLabels(imdsTest.Labels,  classNames);

fprintf('تعداد نمونه‌های Train: %d | تعداد نمونه‌های Test: %d\n', ...
    numel(trainLabels), numel(testLabels));
disp('توزیع کلاس‌ها در مجموعه آموزش (Train):');
disp(countEachLabel(imdsTrain));

%%
fprintf('--> در حال بارگذاری EfficientNetB0...\n');
try
    netEff = efficientnetb0;
catch ME
    error(['بارگذاری EfficientNetB0 ناموفق بود. Add-on مربوطه را نصب کنید.\n%s'], ME.message);
end

%%
fprintf('\n--> استخراج ویژگی‌ها از ResNet50...\n');
featTrainRes = extractNetworkFeatures(netResNet, imdsTrain);
featTestRes  = extractNetworkFeatures(netResNet, imdsTest);

fprintf('--> استخراج ویژگی‌ها از DenseNet201...\n');
featTrainDense = extractNetworkFeatures(netDenseNet, imdsTrain);
featTestDense  = extractNetworkFeatures(netDenseNet, imdsTest);

fprintf('--> استخراج ویژگی‌ها از EfficientNetB0...\n');
featTrainEff = extractNetworkFeatures(netEff, imdsTrain);
featTestEff  = extractNetworkFeatures(netEff, imdsTest);

%%
fprintf('--> فیوژن بردارهای ویژگی...\n');
XTrain = [featTrainRes, featTrainDense, featTrainEff];
XTest  = [featTestRes,  featTestDense,  featTestEff];
fprintf('طول بردار ویژگی نهایی: %d بعد\n', size(XTrain, 2));

%%
fprintf('--> اعمال Z-Score و کاهش بعد با PCA (حفظ ۹۸.۵٪ واریانس)...\n');
[ZTrain, ~, ZTest, numComponents] = fitPcaTransform(XTrain, [], XTest, 98.5);
fprintf('کاهش ابعاد: از %d به %d مؤلفه اصلی.\n', size(XTrain, 2), numComponents);

%%
fprintf('\n--> آموزش متاکلاسیفایر چندکلاسه SVM (One-vs-All)...\n');

bestBoxConstraint = 36.623;
bestKernelScale    = 69.115;
bestPolyOrder      = 3;

template = templateSVM( ...
    'KernelFunction',   'polynomial', ...
    'PolynomialOrder',  bestPolyOrder, ...
    'BoxConstraint',    bestBoxConstraint, ...
    'KernelScale',      bestKernelScale, ...
    'Standardize',      true);

finalModel = fitcecoc(ZTrain, trainLabels, ...
    'Learners', template, ...
    'Coding',   'onevsall', ...
    'Verbose',  0);

%%
[predOpt, ~] = predict(finalModel, ZTest);
accOpt = sum(predOpt == testLabels) / numel(testLabels) * 100;

fprintf('\n================ نتیجه نهایی تست ================\n');
fprintf('دقت کلی (Overall Accuracy): %.2f%%\n\n', accOpt);

for c = 1:numel(classNames)
    idx = (testLabels == classNames{c});
    classAcc = sum(predOpt(idx) == testLabels(idx)) / sum(idx) * 100;
    fprintf('دقت کلاس %-12s: %.2f%%\n', classNames{c}, classAcc);
end
fprintf('===============================================\n');


%%
% Helper Functions
% ========================================================================

function y = normalizeLabels(y, classNames)
    y = categorical(cellstr(string(y)), classNames);
end

function features = extractNetworkFeatures(net, imds)
    inputSize = net.Layers(1).InputSize(1:2);
    featureLayer = findFeatureLayer(net);

    ads = augmentedImageDatastore(inputSize, imds, 'ColorPreprocessing', 'gray2rgb');
    features = activations(net, ads, featureLayer, 'OutputAs', 'rows');
    features = double(features);
end

function layerName = findFeatureLayer(net)
    layers = net.Layers;
    layerName = '';

    for i = 1:numel(layers)
        layerClass = lower(class(layers(i)));
        currentName = lower(char(layers(i).Name));

        isAveragePool = contains(layerClass, 'averagepool') || ...
                        contains(layerClass, 'globalaveragepool');

        isKnownAveragePoolName = contains(currentName, 'avg_pool') || ...
                                 contains(currentName, 'global_average_pool');

        if isAveragePool || isKnownAveragePoolName
            layerName = layers(i).Name;
        end
    end

    if isempty(layerName)
        fcIndex = [];
        for i = 1:numel(layers)
            if contains(lower(class(layers(i))), 'fullyconnected')
                fcIndex = i;
            end
        end

        if ~isempty(fcIndex) && fcIndex > 1
            layerName = layers(fcIndex - 1).Name;
        elseif numel(layers) >= 3
            layerName = layers(end - 2).Name;
        else
            error('لایه مناسب برای استخراج ویژگی پیدا نشد.');
        end
    end
end

function [ZTrain, ZB, ZC, numComponents] = fitPcaTransform(XTrain, XB, XC, targetVariance)
    mu    = mean(XTrain, 1);
    sigma = std(XTrain, 0, 1);
    sigma(~isfinite(sigma) | sigma == 0) = 1;

    XTrainScaled = (XTrain - mu) ./ sigma;

    [coeff, ~, ~, ~, explained, pcaMu] = pca(XTrainScaled, 'Economy', true);

    cumulativeExplained = cumsum(explained);
    numComponents = find(cumulativeExplained >= targetVariance, 1);

    if isempty(numComponents)
        numComponents = size(coeff, 2);
    end

    coeff  = coeff(:, 1:numComponents);
    ZTrain = (XTrainScaled - pcaMu) * coeff;

    if ~isempty(XB)
        XBScaled = (XB - mu) ./ sigma;
        ZB = (XBScaled - pcaMu) * coeff;
    else
        ZB = [];
    end

    if ~isempty(XC)
        XCScaled = (XC - mu) ./ sigma;
        ZC = (XCScaled - pcaMu) * coeff;
    else
        ZC = [];
    end
end