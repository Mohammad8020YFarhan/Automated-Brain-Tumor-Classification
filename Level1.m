%% ========================================================================

clc;
clear;
close all;

%% 1.

rootDir = 'E:\';
trainDir = fullfile(rootDir, 'Training');
testDir  = fullfile(rootDir, 'Testing');

if ~exist(trainDir, 'dir') || ~exist(testDir, 'dir')
    error(['مسیرهای Training یا Testing در دایرکتوری T یافت نشدند. ' ...
           'لطفاً مسیر دیتاست را بررسی کنید.']);
end

imdsTrain = imageDatastore(trainDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsTest = imageDatastore(testDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

classNames = categories(imdsTrain.Labels);
numClasses = numel(classNames);

if numClasses ~= 4
    warning('تعداد کلاس‌های شناسایی‌شده برابر 4 نیست.');
end

fprintf('========================================\n');
fprintf('تعداد کلاس‌های شناسایی‌شده: %d\n', numClasses);
disp(classNames);
fprintf('تعداد تصاویر آموزشی: %d\n', numel(imdsTrain.Files));
fprintf('تعداد تصاویر آزمایشی: %d\n', numel(imdsTest.Files));
fprintf('========================================\n');

%% 2.

inputSize = [224 224 3];

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandRotation', [-15 15], ...
    'RandXScale', [0.9 1.1], ...
    'RandYScale', [0.9 1.1]);

augImdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augImdsTest = augmentedImageDatastore(inputSize, imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

%% ========================================================================
%% ========================================================================

fprintf('\n========================================\n');
fprintf('شروع آموزش مدل ResNet-50\n');
fprintf('========================================\n');

%% 3.

net = resnet50;
lgraph = layerGraph(net);

lgraph = removeLayers(lgraph, ...
    {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'});

newLayers = [
    fullyConnectedLayer(numClasses, ...
        'Name', 'new_fc', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10)

    softmaxLayer('Name', 'new_softmax')

    classificationLayer('Name', 'new_classoutput')
];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'avg_pool', 'new_fc');

%% 4.

optionsResNet = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 1e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 6, ...
    'L2Regularization', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augImdsTest, ...
    'ValidationFrequency', max(1, floor(numel(imdsTrain.Files) / 32)), ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

%% 5.

fprintf('\n>>> در حال آموزش ResNet-50...\n');

[trainedNetResNet, infoResNet] = trainNetwork( ...
    augImdsTrain, lgraph, optionsResNet);

fprintf('>>> آموزش ResNet-50 با موفقیت پایان یافت.\n');

%% 6.

fprintf('\n>>> ارزیابی ResNet-50 روی داده‌های Testing...\n');

[YPredResNet, probsResNet] = classify( ...
    trainedNetResNet, augImdsTest);

YTest = standardizeLabels(imdsTest.Labels, classNames);
YPredResNet = standardizeLabels(YPredResNet, classNames);

[metricsResNet, summaryResNet, cmResNet] = calculateMetrics( ...
    YTest, YPredResNet, classNames);

fprintf('\n========================================\n');
fprintf('نتایج ResNet-50\n');
fprintf('========================================\n');

disp(metricsResNet);
disp(summaryResNet);

fprintf('Accuracy ResNet-50: %.2f%%\n', ...
    summaryResNet.Accuracy * 100);

%% 7.

plotNumericConfusionMatrix( ...
    cmResNet, classNames, ...
    sprintf('ResNet-50 Confusion Matrix - Accuracy: %.2f%%', ...
    summaryResNet.Accuracy * 100));

%% 8.

plotTrainingHistory(infoResNet, 'ResNet-50');

%% ========================================================================
%% ========================================================================

fprintf('\n========================================\n');
fprintf('شروع آموزش مدل دوم DenseNet-201\n');
fprintf('========================================\n');

if exist('densenet201', 'file') ~= 2
    error(['تابع densenet201 در MATLAB یافت نشد. ' ...
           'پشتیبانی DenseNet-201 را از Add-On Explorer نصب کنید.']);
end

clear net lgraph newLayers;

%% 9.

augImdsTrainDenseNet = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augImdsTestDenseNet = augmentedImageDatastore(inputSize, imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

%% 10.

netDenseNet = densenet201;
lgraphDenseNet = layerGraph(netDenseNet);

lgraphDenseNet = removeLayers(lgraphDenseNet, ...
    {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'});

newLayersDenseNet = [
    fullyConnectedLayer(numClasses, ...
        'Name', 'new_fc_densenet', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10)

    softmaxLayer('Name', 'new_softmax_densenet')

    classificationLayer('Name', 'new_classoutput_densenet')
];

lgraphDenseNet = addLayers(lgraphDenseNet, newLayersDenseNet);
lgraphDenseNet = connectLayers( ...
    lgraphDenseNet, 'avg_pool', 'new_fc_densenet');

%% 11.

optionsDenseNet = trainingOptions('adam', ...
    'MiniBatchSize', 16, ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 1e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 6, ...
    'L2Regularization', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augImdsTestDenseNet, ...
    'ValidationFrequency', max(1, floor(numel(imdsTrain.Files) / 16)), ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

%% 12.

fprintf('\n>>> در حال آموزش DenseNet-201...\n');

[trainedNetDenseNet, infoDenseNet] = trainNetwork( ...
    augImdsTrainDenseNet, ...
    lgraphDenseNet, ...
    optionsDenseNet);

fprintf('>>> آموزش DenseNet-201 با موفقیت پایان یافت.\n');

%% 13.

fprintf('\n>>> ارزیابی DenseNet-201 روی داده‌های Testing...\n');

[YPredDenseNet, probsDenseNet] = classify( ...
    trainedNetDenseNet, augImdsTestDenseNet);

YPredDenseNet = standardizeLabels(YPredDenseNet, classNames);

[metricsDenseNet, summaryDenseNet, cmDenseNet] = calculateMetrics( ...
    YTest, YPredDenseNet, classNames);

fprintf('\n========================================\n');
fprintf('نتایج DenseNet-201\n');
fprintf('========================================\n');

disp(metricsDenseNet);
disp(summaryDenseNet);

fprintf('Accuracy DenseNet-201: %.2f%%\n', ...
    summaryDenseNet.Accuracy * 100);

%% 14.

plotNumericConfusionMatrix( ...
    cmDenseNet, classNames, ...
    sprintf('DenseNet-201 Confusion Matrix - Accuracy: %.2f%%', ...
    summaryDenseNet.Accuracy * 100));

%% 15.

plotTrainingHistory(infoDenseNet, 'DenseNet-201');

%% ========================================================================
%% ========================================================================

fprintf('\n========================================\n');
fprintf('ایجاد مدل ترکیبی Ensemble\n');
fprintf('========================================\n');

probsResNetAligned = alignProbabilities( ...
    probsResNet, trainedNetResNet, classNames);

probsDenseNetAligned = alignProbabilities( ...
    probsDenseNet, trainedNetDenseNet, classNames);

weightResNet = 0.50;
weightDenseNet = 0.50;

ensembleProbabilities = ...
    weightResNet * probsResNetAligned + ...
    weightDenseNet * probsDenseNetAligned;

[~, predictedIndices] = max(ensembleProbabilities, [], 2);

YPredEnsemble = categorical( ...
    classNames(predictedIndices), classNames);

%% 16.

[metricsEnsemble, summaryEnsemble, cmEnsemble] = calculateMetrics( ...
    YTest, YPredEnsemble, classNames);

fprintf('\n========================================\n');
fprintf('نتایج نهایی مدل ترکیبی Ensemble\n');
fprintf('========================================\n');

disp(metricsEnsemble);
disp(summaryEnsemble);

fprintf('Accuracy Ensemble: %.2f%%\n', ...
    summaryEnsemble.Accuracy * 100);

fprintf('Recall Macro: %.2f%%\n', ...
    summaryEnsemble.Recall * 100);

fprintf('Specificity Macro: %.2f%%\n', ...
    summaryEnsemble.Specificity * 100);

fprintf('Precision Macro: %.2f%%\n', ...
    summaryEnsemble.Precision * 100);

fprintf('F1-Score Macro: %.2f%%\n', ...
    summaryEnsemble.F1Score * 100);

%% 17.

plotNumericConfusionMatrix( ...
    cmEnsemble, classNames, ...
    sprintf('Ensemble Confusion Matrix - Accuracy: %.2f%%', ...
    summaryEnsemble.Accuracy * 100));

%% 18.

modelNames = categorical( ...
    {'ResNet-50', 'DenseNet-201', 'Ensemble'}, ...
    {'ResNet-50', 'DenseNet-201', 'Ensemble'});

accuracyValues = [
    summaryResNet.Accuracy * 100
    summaryDenseNet.Accuracy * 100
    summaryEnsemble.Accuracy * 100
];

figure('Color', 'w', 'Position', [200 200 850 550]);

bar(modelNames, accuracyValues);

ylabel('Accuracy (%)');
title('Comparison of Classification Accuracy');
ylim([0 100]);
grid on;

for k = 1:numel(accuracyValues)
    text(k, accuracyValues(k) + 1, ...
        sprintf('%.2f%%', accuracyValues(k)), ...
        'HorizontalAlignment', 'center', ...
        'FontWeight', 'bold', ...
        'FontSize', 11);
end

%% 19.

save('BrainTumor_ResNet50_DenseNet201_Ensemble.mat', ...
    'trainedNetResNet', ...
    'trainedNetDenseNet', ...
    'accuracyValues', ...
    'metricsResNet', ...
    'metricsDenseNet', ...
    'metricsEnsemble', ...
    'summaryResNet', ...
    'summaryDenseNet', ...
    'summaryEnsemble', ...
    'cmResNet', ...
    'cmDenseNet', ...
    'cmEnsemble', ...
    'classNames');

fprintf('\n========================================\n');
fprintf('[+] تمام مدل‌ها و نتایج ذخیره شدند.\n');
fprintf('نام فایل ذخیره‌شده:\n');
fprintf('BrainTumor_ResNet50_DenseNet201_Ensemble.mat\n');
fprintf('========================================\n');


%% ========================================================================
%% ========================================================================

function labels = standardizeLabels(labels, classNames)

    labels = categorical(cellstr(labels), classNames);

end


function [metricsTable, summaryTable, C] = calculateMetrics( ...
    YTrue, YPred, classNames)

    classOrder = categorical(classNames, classNames);

    YTrue = categorical(cellstr(YTrue), classNames);
    YPred = categorical(cellstr(YPred), classNames);

    C = confusionmat(YTrue, YPred, ...
        'Order', classOrder);

    totalSamples = sum(C(:));

    truePositive = diag(C);

    falseNegative = sum(C, 2) - truePositive;
    falsePositive = sum(C, 1)' - truePositive;

    trueNegative = totalSamples ...
        - truePositive ...
        - falseNegative ...
        - falsePositive;

    recall = safeDivide( ...
        truePositive, truePositive + falseNegative);

    specificity = safeDivide( ...
        trueNegative, trueNegative + falsePositive);

    precision = safeDivide( ...
        truePositive, truePositive + falsePositive);

    f1Score = safeDivide( ...
        2 .* precision .* recall, precision + recall);

    accuracy = sum(truePositive) / totalSamples;

    metricsTable = table( ...
        classOrder, ...
        truePositive, ...
        falseNegative, ...
        falsePositive, ...
        trueNegative, ...
        precision, ...
        recall, ...
        specificity, ...
        f1Score, ...
        'VariableNames', { ...
        'Class', ...
        'TP', ...
        'FN', ...
        'FP', ...
        'TN', ...
        'Precision', ...
        'Recall', ...
        'Specificity', ...
        'F1Score'});

    summaryTable = table( ...
        accuracy, ...
        mean(recall), ...
        mean(specificity), ...
        mean(precision), ...
        mean(f1Score), ...
        'VariableNames', { ...
        'Accuracy', ...
        'Recall', ...
        'Specificity', ...
        'Precision', ...
        'F1Score'});

end


function result = safeDivide(numerator, denominator)

    result = zeros(size(numerator));

    validIndices = denominator ~= 0;

    result(validIndices) = ...
        numerator(validIndices) ./ denominator(validIndices);

end


function probsAligned = alignProbabilities( ...
    probabilities, trainedNetwork, classNames)

    finalLayer = trainedNetwork.Layers(end);

    networkClassNames = cellstr(finalLayer.Classes);

    [isFound, locations] = ismember( ...
        classNames, networkClassNames);

    if ~all(isFound)
        error('ترتیب کلاس‌های خروجی شبکه با کلاس‌های دیتاست هماهنگ نیست.');
    end

    probsAligned = probabilities(:, locations);

end


function plotNumericConfusionMatrix( ...
    C, classNames, figureTitle)

    figure('Color', 'w', ...
        'Position', [100 100 850 700]);

    imagesc(C);

    axis image;
    colorbar;
    colormap(parula(256));

    title(figureTitle, ...
        'FontWeight', 'bold', ...
        'FontSize', 13);

    xlabel('Predicted Class', 'FontWeight', 'bold');
    ylabel('True Class', 'FontWeight', 'bold');

    xticks(1:numel(classNames));
    yticks(1:numel(classNames));

    xticklabels(classNames);
    yticklabels(classNames);

    xtickangle(35);

    maxValue = max(C(:));

    if maxValue == 0
        maxValue = 1;
    end

    for row = 1:size(C, 1)
        for col = 1:size(C, 2)

            if C(row, col) > maxValue / 2
                textColor = 'w';
            else
                textColor = 'k';
            end

            text(col, row, sprintf('%d', C(row, col)), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'Color', textColor, ...
                'FontSize', 13, ...
                'FontWeight', 'bold');
        end
    end

    set(gca, ...
        'FontSize', 11, ...
        'LineWidth', 1.1);

end


function plotTrainingHistory(trainingInfo, modelName)

    if ~isstruct(trainingInfo)
        warning('اطلاعات آموزش برای %s قابل رسم نیست.', modelName);
        return;
    end

    if ~isfield(trainingInfo, 'Iteration')
        warning('فیلد Iteration برای %s وجود ندارد.', modelName);
        return;
    end

    iteration = trainingInfo.Iteration;

    figure('Color', 'w', ...
        'Position', [150 150 1000 700]);

    tiledlayout(2, 1);

    nexttile;

    hold on;

    if isfield(trainingInfo, 'TrainingLoss')

        trainingLoss = trainingInfo.TrainingLoss;
        validTraining = ~isnan(trainingLoss);

        plot(iteration(validTraining), ...
            trainingLoss(validTraining), ...
            'LineWidth', 1.5, ...
            'Color', [0.85 0.15 0.10]);

    end

    if isfield(trainingInfo, 'ValidationLoss')

        validationLoss = trainingInfo.ValidationLoss;
        validValidation = ~isnan(validationLoss);

        plot(iteration(validValidation), ...
            validationLoss(validValidation), ...
            '--', ...
            'LineWidth', 1.5, ...
            'Color', [0.10 0.25 0.80]);

    end

    grid on;
    xlabel('Iteration');
    ylabel('Loss');
    title([modelName ' - Training and Validation Loss']);
    legend('Training Loss', 'Validation Loss', ...
        'Location', 'best');

    hold off;

    nexttile;

    hold on;

    if isfield(trainingInfo, 'TrainingAccuracy')

        trainingAccuracy = trainingInfo.TrainingAccuracy;
        validTrainingAccuracy = ~isnan(trainingAccuracy);

        plot(iteration(validTrainingAccuracy), ...
            trainingAccuracy(validTrainingAccuracy), ...
            'LineWidth', 1.5, ...
            'Color', [0.10 0.55 0.20]);

    end

    if isfield(trainingInfo, 'ValidationAccuracy')

        validationAccuracy = trainingInfo.ValidationAccuracy;
        validValidationAccuracy = ~isnan(validationAccuracy);

        plot(iteration(validValidationAccuracy), ...
            validationAccuracy(validValidationAccuracy), ...
            '--', ...
            'LineWidth', 1.5, ...
            'Color', [0.65 0.10 0.65]);

    end

    grid on;
    xlabel('Iteration');
    ylabel('Accuracy (%)');
    title([modelName ' - Training and Validation Accuracy']);
    legend('Training Accuracy', 'Validation Accuracy', ...
        'Location', 'best');

    ylim([0 100]);

    hold off;

end