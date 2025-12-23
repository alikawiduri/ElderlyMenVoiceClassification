%% =========================================================
% EVALUATION VISUALIZATION
% Spectrogram, MFCC, and Pitch Analysis
% =========================================================

clear; clc;

%% Load metadata
load('data/clean_metadata.mat','T_final');

fsTarget = 16000;

% Ambil 1 contoh elderly & non-elderly
idx_elderly = find(T_final.label=="elderly",1);
idx_non     = find(T_final.label=="not_elderly",1);

files = { ...
    fullfile('data',T_final.filename{idx_elderly}), ...
    fullfile('data',T_final.filename{idx_non}) ...
};

labels = {'elderly','non_elderly'};   % aman untuk nama file

%% Loop visualisasi
for k = 1:2

    %% Load audio
    [audio,fs] = audioread(files{k});
    audio = mean(audio,2);                         % mono
    audio = resample(audio,fsTarget,fs);
    audio = audio ./ (max(abs(audio)) + eps);

    %% Trim silence
    idx = find(abs(audio) > 0.01);
    audio = audio(idx(1):idx(end));

    %% =======================
    %% 1. Spectrogram
    %% =======================
    fig = figure('Visible','off');
    spectrogram(audio, ...
        round(0.025*fsTarget), ...
        round(0.015*fsTarget), ...
        1024, fsTarget, 'yaxis');
    title(['Spectrogram - ' labels{k}]);
    colormap jet; colorbar;

    exportgraphics(fig, ...
        fullfile('result',['spectrogram_' labels{k} '.png']), ...
        'Resolution',300);
    close(fig);

    %% =======================
    %% 2. MFCC Heatmap
    %% =======================
    frameLength = round(0.025*fsTarget);
    win = hamming(frameLength,'periodic');

    c = mfcc(audio,fsTarget,...
        'NumCoeffs',13,...
        'Window',win,...
        'OverlapLength',round(0.015*fsTarget));

    fig = figure('Visible','off');
    imagesc(c');
    axis xy;
    colormap jet; colorbar;
    xlabel('Frame Index');
    ylabel('MFCC Coefficient');
    title(['MFCC Heatmap - ' labels{k}]);

    exportgraphics(fig, ...
        fullfile('result',['mfcc_' labels{k} '.png']), ...
        'Resolution',300);
    close(fig);

    %% =======================
    %% 3. Pitch Contour
    %% =======================
    f0 = pitch(audio,fsTarget,'Range',[50 250]);

    fig = figure('Visible','off');
    plot(f0,'LineWidth',1.5);
    xlabel('Frame Index');
    ylabel('Pitch (Hz)');
    title(['Pitch Contour - ' labels{k}]);
    grid on;

    exportgraphics(fig, ...
        fullfile('result',['pitch_' labels{k} '.png']), ...
        'Resolution',300);
    close(fig);

end

disp('Evaluation visualization completed and saved to result/.');
