%% Load metadata
T = readtable('data/cv-valid-train.csv');

%% Filter male only
T = T(strcmp(T.gender,'male'),:);

%% Remove missing age
T = T(~ismissing(T.age),:);

%% Define elderly label
elderlyAges = {'sixties','seventies','eighties','nineties'};
T.label = repmat("not_elderly",height(T),1);
T.label(ismember(T.age,elderlyAges)) = "elderly";

%% Keep only needed columns (LABEL HARUS IKUT)
T = T(:,{'filename','age','gender','duration','label'});

disp(countcats(categorical(T.label)))

%% Subsampling agar seimbang dan ringan
rng(1); % supaya hasil konsisten

T_elderly = T(T.label=="elderly",:);
T_non     = T(T.label=="not_elderly",:);

% Tentukan jumlah maksimum per kelas
N = min([height(T_elderly), height(T_non), 500]);

T_elderly = T_elderly(randperm(height(T_elderly), N), :);
T_non     = T_non(randperm(height(T_non), N), :);

T_final = [T_elderly; T_non];

disp("Final dataset size:")
disp(countcats(categorical(T_final.label)))

save('data/clean_metadata.mat','T_final')

file = fullfile('data', T_final.filename{1});
exist(file,'file')

