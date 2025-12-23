load('data/clean_metadata.mat','T_final');

fsTarget = 16000;
frameLength = round(0.025 * fsTarget);   % 25 ms
hopLength   = round(0.010 * fsTarget);   % 10 ms

% 🔴 WINDOW HARUS DIDEFINISIKAN
win = hamming(frameLength,'periodic');

X = [];
Y = [];

for i = 1:height(T_final)

    file = fullfile('data', T_final.filename{i});
    if ~isfile(file)
        continue;
    end

    % --- Read audio ---
    [audio,fs] = audioread(file);
    audio = mean(audio,2);                        % mono
    audio = resample(audio,fsTarget,fs);          % resample
    audio = audio ./ (max(abs(audio)) + eps);     % normalize

    % --- Trim silence ---
    idx = find(abs(audio) > 0.01);
    if isempty(idx)
        continue;
    end
    audio = audio(idx(1):idx(end));

    % --- Duration check ---
    dur = numel(audio) / fsTarget;
    if dur < 0.5 || dur > 10
        continue;
    end

% --- MFCC ---
    c = mfcc(audio, fsTarget, ...
    'NumCoeffs', 13, ...
    'Window', win, ...
    'OverlapLength', frameLength - hopLength);

    d  = diff(c);
    dd = diff(d);

% --- Prosodic features ---
% Pitch (F0)
    f0 = pitch(audio, fsTarget);
    f0 = f0(~isnan(f0));   % buang NaN

    if isempty(f0)
        f0_mean = 0;
        f0_std  = 0;
    else
        f0_mean = mean(f0);
        f0_std  = std(f0);
    end

% Energy
    energy = rms(audio);

% Duration
    dur = numel(audio) / fsTarget;

% --- Pooling ---
    feat = [ ...
        mean(c,1),  std(c,[],1), ...
        mean(d,1),  std(d,[],1), ...
        mean(dd,1), std(dd,[],1), ...
        f0_mean, f0_std, energy, dur ...
    ];

    X = [X; feat];
    Y = [Y; T_final.label(i)];

end

Y = categorical(Y);

save('result/features.mat','X','Y');

disp('Feature matrix size:');
disp(size(X));

disp('Class distribution:');
disp(countcats(Y));
