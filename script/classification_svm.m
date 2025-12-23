%% Load fitur
load('result/features.mat','X','Y');
Y = categorical(Y);

%% Split data (80% train, 20% test)
cv = cvpartition(Y,'HoldOut',0.2);

X_train = X(training(cv),:);
Y_train = Y(training(cv));
X_test  = X(test(cv),:);
Y_test  = Y(test(cv));

%% Standardisasi (FIT di TRAIN)
[X_train_z, mu, sigma] = zscore(X_train);
X_test_z = (X_test - mu) ./ sigma;

%% PCA (FIT di TRAIN, retain 95% variance)
[coeff, scoreTrain, ~, ~, explained] = pca(X_train_z);
k = find(cumsum(explained) >= 95, 1);

X_train_pca = scoreTrain(:,1:k);
X_test_pca  = X_test_z * coeff(:,1:k);

%% Train SVM (model FINAL)
svmModel = fitcsvm( ...
    X_train_pca, Y_train, ...
    'KernelFunction','rbf', ...
    'Standardize',false, ... % sudah distandardisasi
    'OptimizeHyperparameters',{'BoxConstraint','KernelScale'}, ...
    'HyperparameterOptimizationOptions',struct( ...
        'Verbose',0));

%% Prediksi
Y_pred = predict(svmModel, X_test_pca);

%% Evaluasi & SIMPAN CONFUSION MATRIX
fig = figure('Visible','off');
confusionchart(Y_test, Y_pred);
title('Confusion Matrix - SVM (Final Model)');

exportgraphics(fig, ...
    fullfile('result','confusion_matrix_svm.png'), ...
    'Resolution',300);
close(fig);

accuracy = mean(Y_pred == Y_test);
fprintf('Final SVM Accuracy: %.2f %%\n', accuracy*100);

%% Simpan model dan parameter preprocessing
model.svmModel = svmModel;
model.mu       = mu;
model.sigma    = sigma;
model.coeff    = coeff(:,1:k);
model.k        = k;

save('result/svm_final_model.mat','model');

disp('Model SVM final & confusion matrix berhasil disimpan di result/');
