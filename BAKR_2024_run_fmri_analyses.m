%
% Fmri analyses for Buergi, Aydogan, Konovalov, & Ruff (2024):
% "A neural fingerprint of adaptive mentalization" 
%
% Re-creates the outputs in the results folder based on preprocessed data
% (To simply create figures and run statistical analyses, use the script
% BAKR_2024_results_and_figures instead)
%

%% set folder

% project_folder = fileparts(mfilename('fullpath'));
% project_folder = 'E:\2024_CHASE';
project_folder = cd;
addpath(fullfile(project_folder,'source'));

%% --------------------------------------------------------------------------- %
%                              PRIMARY DATASET                                 %
% --------------------------------------------------------------------------- %%

%% get data

folders.project = project_folder;
folders.prepro = fullfile(project_folder,'data','fmri','primary');
folders.results = fullfile(project_folder,'results');

folders.spm = fileparts(which('spm'));

% get subject names/folders
subj_files = dir(fullfile(folders.prepro, 'sub-00*'));
subj_folders = subj_files([subj_files.isdir]');
subjects = {subj_folders.name}';

% exclude subjects meeting exclusion criteria
subjects(strcmp(subjects,'sub-00006')) = []; % SUBID 7
subjects(strcmp(subjects,'sub-00087')) = []; % SUBID 94

% specify model
config.folders = folders;
config.dataset = 'primary';

%% univariate

% further specifications
config.type    = 'pmods_whole_run';
config.pmods   = {'SV_ch','subj_KL_div_fb','APE_fb','SV_fb','reward_fb'};

% subject level
parfor i_subj = 1:numel(subjects)
    BAKR_2024_1L_model(config,subjects{i_subj});
    BAKR_2024_1L_con(config,subjects{i_subj},0);
end

% group level
config.voi = {'socialbrain'};
config.stat_models = {'ttest'};
con_names = BAKR_2024_1L_con(config,subjects{1},1);
parfor i_con = 3:5
    BAKR_2024_2L(config,subjects,con_names,i_con);
end   

% robustness check: controlling for effects of time
control_models = {'time_first_order','time_second_order'};
pmods = {'SV_ch','subj_KL_div_fb','APE_fb','SV_fb','reward_fb','trial_fb','trial_2_fb'};
for i_model = 1:numel(control_models)
    config.folders.results = fullfile(folders.results,'control_analyses',control_models{i_model});
    config.pmods = pmods(1:5+i_model);
    parfor i_subj = 1:numel(subjects)
        BAKR_2024_1L_model(config,subjects{i_subj});
        BAKR_2024_1L_con(config,subjects{i_subj},0);
    end
    config.voi = {'previous_clusters'};
    BAKR_2024_2L(config,subjects,con_names,4);  
end
config.folders.results = folders.results;

%% multivariate: levels

% assigning whole runs to most often played level
config.type  = 'levels_whole_run';
config.pmods = 'none';
parfor i_subj = 1:numel(subjects)
    BAKR_2024_1L_model(config,subjects{i_subj});
end
BAKR_2024_decode_levels(config,subjects);

% only using trials that can be assigned to a level with certainty (i.e. belief > .5)
config.type  = 'levels_thresholded';
config.pmods = 'subj_p_k'; 
parfor i_subj = 1:numel(subjects)
    BAKR_2024_1L_model(config,subjects{i_subj});
end
BAKR_2024_decode_levels(config,subjects);

%% multivariate: belief update

% specify model
config.type  = 'pmod_ordinal';
config.pmods = 'subj_KL_div_fb';

% first-level models
parfor i_subj = 1:numel(subjects)
    BAKR_2024_1L_model(config,subjects{i_subj});
end

% MVPA decoding
BAKR_2024_decode_BU(config,subjects,'train');

% control analysis: decoding based on average ROI activity
BAKR_2024_decode_BU(config,subjects,'train_control');

% save participants (for cross-decoding)
config_primary = config;
subjects_primary = subjects;

%% --------------------------------------------------------------------------- %
%                                 REPLICATION                                  %
% --------------------------------------------------------------------------- %%

%% get data

folders.prepro = fullfile(project_folder,'data','fmri','replication');
folders.results = fullfile(project_folder,'results','replication');

folders.spm = fileparts(which('spm'));

% get subject names/folders
subj_files = dir(fullfile(folders.prepro, 'sub-*'));
subj_folders = subj_files([subj_files.isdir]');
subjects = {subj_folders.name}';

% exclude subjects meeting exclusion criteria
subjects(strcmp(subjects,'sub-00084')) = [];

config.folders = folders;
config.dataset = 'replication';

%% univariate

config.type = 'pmods_whole_run';
config.pmods = {'SV_ch','subj_KL_div_fb','APE_fb','SV_fb','reward_fb'};
parfor i_subj = 1:numel(subjects)
    BAKR_2024_1L_model(config,subjects{i_subj});
    BAKR_2024_1L_con(config,subjects{i_subj},0);
end

config.voi = {'previous_clusters'};
config.stat_models = {'ttest'};
con_names = BAKR_2024_1L_con(config,subjects{1},1);
parfor i_con = 3:5
    BAKR_2024_2L(config,subjects,con_names,i_con);
end   

%% multivariate

config.type  = 'pmod_ordinal';
config.pmods = 'subj_KL_div_fb';

parfor i_subj = 1:numel(subjects)
    BAKR_2024_1L_model(config,subjects{i_subj});
end
BAKR_2024_decode_BU(config,subjects,'predict');

%% --------------------------------------------------------------------------- %
%                                SUPPLEMENTARY                                 %
% --------------------------------------------------------------------------- %%

%% belief update decoding in reverse order

% specify models
[config.type,config_primary.type] = deal('pmod_ordinal');
[config.pmods,config_primary.pmods] = deal('subj_KL_div_fb');

% train on replication dataset, test on primary dataset
BAKR_2024_decode_BU(config,subjects,'train_reverse');
BAKR_2024_decode_BU(config_primary,subjects_primary,'predict_reverse');
