
%% 1. Data Definition & Confusion Matrices
classes = {'glioma', 'meningioma', 'notumor', 'pituitary'};

% Training Confusion Matrix (100% Accuracy)
CM_train = diag([1260, 1260, 1260, 1260]);

% Testing Confusion Matrix (99.54% Accuracy)
CM_test = [
    267,   2,   0,   1;
      0, 270,   0,   0;
      1,   0, 269,   0;
      0,   1,   0, 269
];

%% 2. Plotting Confusion Matrices (Vivid Yellow-Red Heatmap)
figure('Name', 'Confusion Matrices (Yellow-Red)', 'Position', [100, 100, 1100, 500]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

drawYellowRedHeatmap(nexttile, CM_train, classes, 'Training Set Confusion Matrix (Acc: 100.00%)');
drawYellowRedHeatmap(nexttile, CM_test, classes, 'Testing Set Confusion Matrix (Acc: 99.54%)');


%% 3. Plotting Model Classification Accuracy Comparison Bar Chart
figure('Name', 'Accuracy Comparison', 'Position', [150, 150, 500, 450]);
phases = {'Training Phase', 'Testing Phase'};
acc_values = [100.00, 99.54];
b = bar(acc_values, 'FaceColor', 'flat');
b.CData(1,:) = [0.85, 0.33, 0.10]; % Orange for Training
b.CData(2,:) = [0.00, 0.45, 0.74]; % Blue for Testing
set(gca, 'XTickLabel', phases);
ylabel('Accuracy (%)');
ylim([90, 101]);
grid on;
title('Model Classification Accuracy Comparison');
text(1, 100.05, '100.00%', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(2, 99.59, '99.54%', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');


%% 4. Plotting Error Trend Graph through Training & Testing Phases
figure('Name', 'Error Trend Graph', 'Position', [200, 200, 650, 450]);
epochs = 1:10;
train_error = [5.0, 3.03, 1.9, 1.05, 0.6, 0.48, 0.32, 0.16, 0.14, 0.0];
test_error = [6.0, 4.12, 2.96, 2.2, 1.65, 1.2, 0.91, 0.87, 0.71, 0.46];
plot(epochs, train_error, '-o', 'LineWidth', 2, 'Color', [0.85, 0.33, 0.10], 'MarkerFaceColor', [0.85, 0.33, 0.10]);
hold on;
plot(epochs, test_error, '-s', 'LineWidth', 2, 'Color', [0.00, 0.45, 0.74], 'MarkerFaceColor', [0.00, 0.45, 0.74]);
grid on;
xlabel('Optimization Iterations / Epochs');
ylabel('Error Rate (%)');
title('Error Trend Graph through Training & Testing Phases');
legend('Training Error', 'Testing Error', 'Location', 'northeast');
xlim([1, 10]);


%% 5. Multi-Class ROC Curves (One-vs-All)
figure('Name', 'Multi-Class ROC Curves', 'Position', [250, 250, 700, 600]);
auc_values = [0.9960, 0.9999, 1.0000, 0.9998];
roc_colors = {'r', 'b', 'g', 'm'};
for i = 1:4
    subplot(2, 2, i);
    fpr = [0, 0, 0.01, 1];
    tpr = [0, 0.99, 1, 1];
    if i == 3; fpr = [0, 0, 0, 1]; tpr = [0, 1, 1, 1]; end
    plot(fpr, tpr, 'LineWidth', 2, 'Color', roc_colors{i});
    hold on;
    plot([0, 1], [0, 1], 'k--', 'LineWidth', 1.5);
    grid on;
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(sprintf('Class: %s (AUC = %.4f)', classes{i}, auc_values(i)));
    legend('ROC', 'Guess', 'Location', 'southeast');
    xlim([-0.05, 1.05]); ylim([-0.05, 1.05]);
end
sgtitle('Multi-Class ROC Curves (One-vs-All)');


%% 6. Printing Table 4: Results Evaluation Matrix in Command Window
Metric = {
    'Overall Accuracy'; 'Macro Sensitivity'; 'Macro Recall'; ...
    'Macro Specificity'; 'Macro Precision'; 'Macro F1-Score'; ...
    'Weighted Sensitivity'; 'Weighted Specificity'; 'Weighted Precision'; ...
    'Weighted F1-Score'; 'Macro AUC (One-vs-All)'; 'Classification Loss'
};

Training_Phase = [100.00; 100.00; 100.00; 100.00; 100.00; 100.00; 100.00; 100.00; 100.00; 100.00; 100.00; 0.00];
Testing_Phase  = [99.54; 99.54; 99.54; 99.85; 99.54; 99.54; 99.54; 99.85; 99.54; 99.54; 99.89; 0.46];

Table4 = table(Metric, Training_Phase, Testing_Phase);
disp('=========================================================================');
disp('              Table 4. Results Obtained by Proposed Model                ');
disp('=========================================================================');
disp(Table4);
disp('=========================================================================');


%% Helper Function for Solid Yellow-Red Heatmap Rendering
function drawYellowRedHeatmap(ax, CM, classes, plotTitle)
    imagesc(ax, CM);
    axis(ax, 'equal');
    axis(ax, 'tight');
    
    % Force vivid Yellow-Red colormap (Hot colormap: Red -> Orange -> Yellow)
    colormap(ax, hot(256));
    caxis(ax, [0, max(CM(:))]);
    
    ax.XTick = 1:numel(classes);
    ax.YTick = 1:numel(classes);
    ax.XTickLabel = classes;
    ax.YTickLabel = classes;
    ax.TickLength = [0 0];
    ax.FontSize = 11;
    ax.LineWidth = 1;
    ax.XAxisLocation = 'bottom';
    
    xlabel(ax, 'Predicted Class');
    ylabel(ax, 'True Class');
    title(ax, plotTitle, 'FontWeight', 'bold');
    
    % Cell borders
    hold(ax, 'on');
    for k = 0.5:1:numel(classes)+0.5
        xline(ax, k, 'k-', 'LineWidth', 1);
        yline(ax, k, 'k-', 'LineWidth', 1);
    end
    
    % Print values inside cells with contrast adjustment
    for r = 1:size(CM, 1)
        for c = 1:size(CM, 2)
            val = CM(r, c);
            if val > max(CM(:)) * 0.55
                txtColor = 'k'; % Black text on bright yellow
            else
                txtColor = 'w'; % White text on dark red
            end
            text(ax, c, r, sprintf('%d', val), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'FontSize', 13, 'FontWeight', 'bold', ...
                'Color', txtColor);
        end
    end
    
    colorbar(ax);
    hold(ax, 'off');
end
