%
% Model fitting for Buergi, Aydogan, Konovalov, & Ruff (2024):
% "A neural fingerprint of adaptive mentalization" 
%
% Using the MERLIN toolbox (v0.01)
%

%% set folder
project_folder = cd;
addpath(fullfile(project_folder,'source'));
addpath(fullfile(project_folder,'source','MERLIN_toolbox'));
addpath(genpath(fullfile(project_folder, 'VBA-toolbox')));

% Define a separate output directory
output_dir = fullfile(project_folder, 'results', 'llm_subset');

% Empty the output directory if it exists
if exist(output_dir, 'dir')
    fprintf('Emptying output directory: %s\n', output_dir);
    
    % Get all contents of the directory
    contents = dir(output_dir);
    
    % Delete all files and subdirectories
    for i = 1:length(contents)
        if ~strcmp(contents(i).name, '.') && ~strcmp(contents(i).name, '..')
            item_path = fullfile(output_dir, contents(i).name);
            if contents(i).isdir
                % Delete directory and all its contents
                rmdir(item_path, 's');
            else
                % Delete file
                delete(item_path);
            end
        end
    end
else
    fprintf('Output directory does not exist, creating it...\n');
    mkdir(output_dir);
end

% Create output directories if they don't exist
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
if ~exist(fullfile(output_dir, 'replication'), 'dir')
    mkdir(fullfile(output_dir, 'replication'));
end
if ~exist(fullfile(output_dir, 'supplementary'), 'dir')
    mkdir(fullfile(output_dir, 'supplementary'));
end


%% model fitting
% get data
load(fullfile(project_folder,'data','llm_data.mat'));

fprintf('========================================\n');
fprintf('PREPROCESSING LLM DATA\n');
fprintf('========================================\n\n');

%% Step 1: Remove n_trials field if it exists (BAKR doesn't use it in table format)
if ismember('n_trials', data.Properties.VariableNames)
    fprintf('Step 1: Removing n_trials field...\n');
    data.n_trials = [];
end

%% Step 2: Set n_blocks = 1 for all rows (BAKR convention for long-format data)
fprintf('Step 2: Setting n_blocks = 1 (BAKR convention)...\n');
data.n_blocks(:) = 1;

%% Step 3: Verify block structure
fprintf('Step 3: Verifying block structure...\n');
subjects = unique(data.subjID);
for i = 1:numel(subjects)
    idx = (data.subjID == subjects(i));
    blocks = unique(data.block(idx));
    trials_per_block = histcounts(data.block(idx));
    
    fprintf('  Subject %d: %d blocks, %s trials per block\n', ...
        subjects(i), numel(blocks), mat2str(trials_per_block));
end

fprintf('\nPreprocessing complete!\n');
fprintf('========================================\n\n');

%% Convert to struct (MATCHING ORIGINAL BAKR - NO block_var!)
fprintf('Converting to struct format (continuous session, matching original BAKR)...\n');

% Convert WITHOUT block_var - treats all 120 trials as continuous
data = mn_table2struct(data, 'subjID', 'remove_redundancy', ...
                      'exceptions', {'choice_own','choice_other','missing'});

%% Verify conversion
fprintf('\nVerifying struct conversion:\n');
for i = 1:numel(data)
    fprintf('  Subject %d: n_blocks=%d, n_trials=%d, choice_own size=%s\n', ...
        data(i).subjID, data(i).n_blocks, data(i).n_trials, ...
        mat2str(size(data(i).choice_own)));
end

fprintf('\n✓ Data ready for fitting!\n\n');

%% fit CHASE model
model = BAKR_2024_CHASE_config('CH','fitted',3,'RW-freq');
fits = mn_fit(data,model);

% fit alternative static models
models = {BAKR_2024_CHASE_config('LK','fixed',0,'RW-reward'),...  % reinforcement learning
          BAKR_2024_CHASE_config('LK','fixed',0,'EWA-single'),... % self-tuning EWA
          BAKR_2024_CHASE_config('LK','fixed',0,'EWA-full'),...   % standard EWA
          BAKR_2024_CHASE_config('LK','fixed',1,'RW-freq')};      % fictitious play
fits(2:5) = mn_fit(data,models);

% fit alternative adaptive model
model = BAKR_2024_ToMk_config;
fits(6) = mn_fit(data,model);


%% postprocessing
% rename models
fit_labels = arrayfun(@(fit) fit.model.name,fits,'UniformOutput',0);
fits(contains(fit_labels,'_CH_')).model.name             = 'CHASE';
fits(contains(fit_labels,'max-1_RW-freq')).model.name    = 'Ficticious play';
fits(contains(fit_labels,'max-0_RW-reward')).model.name  = 'Reward learner';
fits(contains(fit_labels,'max-0_EWA-single')).model.name = 'Self-tuning EWA';
fits(contains(fit_labels,'max-0_EWA-full')).model.name   = 'Full EWA';

% save fits
MN = fits(1);
save(fullfile(output_dir,'fits_CHASE_struct.mat'),'MN');
save(fullfile(output_dir,'model_comparison.mat'),'fits');

% create output table for CHASE model
data = mn_createOutputTable(fits(1));
data = movevars(data,'alpha','Before','beta');
data = movevars(data,'gamma','Before','lambda');
save(fullfile(output_dir,'fits_CHASE_table.mat'),'data');

%% --------------------------------------------------------------------------- %
%                   POSTERIOR PREDICTIVE SIMULATIONS                           %
% --------------------------------------------------------------------------- %%

fprintf('\n========================================\n');
fprintf('GENERATING POSTERIOR PREDICTIVE SIMULATIONS\n');
fprintf('========================================\n\n');

try
    % Reload data in table format for new_T
    load(fullfile(project_folder,'data','llm_data.mat'));
    if ismember('n_trials', data.Properties.VariableNames)
        data.n_trials = [];
    end
    data.n_blocks(:) = 1;
    new_T = data;  % Save table for plotting
    % clear data;  % Clear to avoid confusion with struct below
    
    % Use all successfully fitted subjects
    idx_valid = find(arrayfun(@(subj) isfield(subj, 'optim') && ~isempty(subj.optim), fits(1).subj));
    fprintf('Using %d subjects with valid fits\n\n', numel(idx_valid));
    
    % Generate simulations for ALL models
    all_sims = [];
    for i_model = 1:numel(fits)
        fprintf('Simulating from model %d/%d: %s\n', i_model, numel(fits), fits(i_model).model.name);
        
        % Simulate data using fitted parameters (no kappa override)
        model_sims = BAKR_2024_simulate_data(fits(i_model), idx_valid);
        
        % Accumulate
        if isempty(all_sims)
            all_sims = model_sims;
        else
            all_sims = [all_sims; model_sims];
        end
        
        fprintf('  Generated %d simulations\n', numel(model_sims));
    end
    
    % Save as 'sims' (required by plotting function)
    sims = all_sims;
    
    % Verification
    fprintf('\nVerification:\n');
    fprintf('  Total simulations: %d\n', numel(sims));
    fprintf('  Models included: %s\n', strjoin(unique({sims.model}), ', '));
    fprintf('  Subjects per model: %d\n', numel(idx_valid));
    fprintf('  Trials per simulation: %d\n', sims(1).n_trials);
    
    % Save both sims and new_T to simulations.mat
    save(fullfile(output_dir,'simulations.mat'), 'sims', 'new_T', '-v7.3');
    
    file_size = dir(fullfile(output_dir,'simulations.mat'));
    fprintf('\n✓ Saved simulations.mat (%d MB)\n', round(file_size.bytes/1024/1024));
    fprintf('  Location: %s\n', fullfile(output_dir,'simulations.mat'));
    
catch e
    warning('Failed to generate posterior predictive simulations: %s', e.message);
    fprintf('  Stack trace:\n');
    for i = 1:length(e.stack)
        fprintf('    [%d] %s (line %d)\n', i, e.stack(i).name, e.stack(i).line);
    end
    fprintf('  Posterior predictive checks will not be available.\n');
end

fprintf('\n========================================\n\n');

%% --------------------------------------------------------------------------- %
%                           SUPPLEMENTARY ANALYSES                             %
% --------------------------------------------------------------------------- %%

fprintf('\n========================================\n');
fprintf('SUPPLEMENTARY ANALYSES\n');
fprintf('========================================\n\n');

%% Identify subjects for supplementary analyses
% Use all subjects that were successfully fit
idx_valid_fits = find(arrayfun(@(subj) isfield(subj, 'optim') && ~isempty(subj.optim), fits(1).subj));

if isempty(idx_valid_fits)
    warning('No successfully fitted subjects found. Skipping supplementary analyses.');
    fprintf('Main fitting complete. Exiting...\n');
    return;
end

fprintf('Using %d subjects for supplementary analyses\n\n', numel(idx_valid_fits));

%% determining the learning rule
% get data (reload from original)
data_for_LR = data;  % data is already in memory from main fitting

% Apply same preprocessing
if ismember('n_trials', data.Properties.VariableNames)
    data.n_trials = [];
end
data.n_blocks(:) = 1;

% Convert WITHOUT block_var (matching main analysis)
data = mn_table2struct(data,'subjID','remove_redundancy',...
                      'exceptions',{'choice_own','choice_other','missing'});

fits_LR(1) = fits(1); % RW-freq already fitted above

% fit alternative versions
clear models
models{1} = BAKR_2024_CHASE_config('CH','fitted',3,'RW-reward');
models{2} = BAKR_2024_CHASE_config('CH','fitted',3,'RW-regret');
models{3} = BAKR_2024_CHASE_config('CH','fitted',3,'RW-hybrid');

try
    fits_LR(2:4) = mn_fit(data, models);
    
    % rename 
    for i_fit = 1:numel(fits_LR)
        fits_LR(i_fit).model.name = fits_LR(i_fit).model.learning_rule(4:end);
    end
    save(fullfile(output_dir,'supplementary','model_comparison_LR.mat'),'fits_LR');
    
    fprintf('✓ Learning rule comparison complete\n\n');
catch e
    warning('Learning rule comparison failed: %s', e.message);
end

%% model recovery: learning rule
try
    fprintf('Running model recovery for learning rules...\n');
    
    % === DIAGNOSTIC: Check fits_LR structure ===
    fprintf('\n=== FITS_LR DIAGNOSTIC ===\n');
    fprintf('Number of models in fits_LR: %d\n', numel(fits_LR));
    for m = 1:numel(fits_LR)
        fprintf('Model %d: %s\n', m, fits_LR(m).model.name);
        fprintf('  Number of subjects: %d\n', numel(fits_LR(m).subj));
        if m == 2 && numel(fits_LR(m).subj) >= 4
            fprintf('  Subject 4 fields: %s\n', strjoin(fieldnames(fits_LR(m).subj(4).data), ', '));
            if isfield(fits_LR(m).subj(4).data, 'bot_level')
                fprintf('  Subject 4 bot_level size: %s\n', mat2str(size(fits_LR(m).subj(4).data.bot_level)));
            end
        end
    end
    fprintf('idx_valid_fits subjects: %s\n', mat2str(idx_valid_fits'));
    fprintf('=== END DIAGNOSTIC ===\n\n');
    
    sims = BAKR_2024_simulate_data(fits_LR, idx_valid_fits);

    % ADD THIS DIAGNOSTIC:
    fprintf('Checking simulated data structure:\n');
    fprintf('  sims(1) type: %s\n', class(sims));
    fprintf('  sims(1) fields: %s\n', strjoin(fieldnames(sims(1)), ', '));
    fprintf('  sims(1).n_trials: %d\n', sims(1).n_trials);
    fprintf('  sims(1).n_blocks: %d\n', sims(1).n_blocks);
    
    models = {fits_LR.model};
    sim_fits = mn_fit(sims', models);
    save(fullfile(output_dir,'supplementary','model_recovery_LR.mat'),'sims','sim_fits');
    
    fprintf('✓ Learning rule model recovery complete\n\n');
catch e
    warning('Model recovery (LR) failed: %s', e.message);
    fprintf('Full error details:\n');
    % BETTER WAY TO DISPLAY STACK:
    for i = 1:length(e.stack)
        fprintf('  [%d] %s (line %d)\n', i, e.stack(i).name, e.stack(i).line);
    end
    fprintf('Identifier: %s\n', e.identifier);
end

%% model recovery: full model
try
    fprintf('Running model recovery for full model...\n');
    sims = BAKR_2024_simulate_data(fits, idx_valid_fits);
    
    models = {fits.model};
    sim_fits = mn_fit(sims', models);
    save(fullfile(output_dir,'model_recovery.mat'),'sims','sim_fits');
    
    fprintf('✓ Full model recovery complete\n\n');
catch e
    warning('Model recovery (full) failed: %s', e.message);
end

%% parameter recovery
try
    fprintf('Running parameter recovery...\n');
    
    [sims,params] = deal([]);
    for k = 0:3
        [new_sims,new_params] = BAKR_2024_simulate_data(fits(1), idx_valid_fits, k);
        params = [params; new_params];
        sims = [sims new_sims];
    end
    
    % fit
    model = fits(1).model;
    fits_est = mn_fit(sims', model);

    estimates = zeros(numel(fits_est.subj), numel(model.params));
    for i = 1:numel(fits_est.subj)
        param_struct = fits_est.subj(i).params;
        for p = 1:numel(model.params)
            estimates(i, p) = param_struct.(model.params(p).name);
        end
    end
    
    prec.model = model;
    prec.params.est = estimates;
    prec.params.gen = params;
    save(fullfile(output_dir,'supplementary','parameter_recovery.mat'),'sims','prec');
    
    fprintf('✓ Parameter recovery complete\n\n');
catch e
    warning('Parameter recovery failed: %s', e.message);
    if ~isempty(e.stack)
        fprintf('  Stack: %s (line %d)\n', e.stack(1).name, e.stack(1).line);
    end
end

%% parameter recovery - BY DATASET

fprintf('\n========================================\n');
fprintf('PARAMETER RECOVERY BY DATASET\n');
fprintf('========================================\n\n');

try
    % DEBUG: Check dataset extraction
    fprintf('\n=== DEBUGGING DATASET EXTRACTION ===\n');
    
    % Step 1: Extract data structs
    data_structs = {fits(1).subj.data};
    fprintf('Number of subjects: %d\n', numel(data_structs));
    fprintf('First data struct fields: %s\n', strjoin(fieldnames(data_structs{1}), ', '));
    
    % Step 2: Check dataset field
    fprintf('\nFirst subject dataset field:\n');
    fprintf('  Class: %s\n', class(data_structs{1}.dataset));
    fprintf('  Size: %s\n', mat2str(size(data_structs{1}.dataset)));
    if iscell(data_structs{1}.dataset)
        fprintf('  Content: {%s}\n', data_structs{1}.dataset{1});
    end
    
    % Step 3: Try extraction
    datasets_raw = cellfun(@(d) d.dataset, data_structs, 'UniformOutput', false);
    fprintf('\nAfter cellfun:\n');
    fprintf('  Class: %s\n', class(datasets_raw));
    fprintf('  Size: %s\n', mat2str(size(datasets_raw)));
    fprintf('  First element class: %s\n', class(datasets_raw{1}));
    if iscell(datasets_raw{1})
        fprintf('  First element content: {%s}\n', datasets_raw{1}{1});
    end
    
    fprintf('=== END DEBUG ===\n\n');
    % Extract all dataset fields
    datasets_raw = cellfun(@(d) d.dataset, {fits(1).subj.data}, 'UniformOutput', false);
    
    % Flatten nested cells: {{'HUMAN'}, {'GPT'}} → {'HUMAN'; 'GPT'}
    datasets = unique(vertcat(datasets_raw{:}));
    
    fprintf('Running parameter recovery for %d datasets:\n', numel(datasets));
    for i = 1:numel(datasets)
        fprintf('  %d. %s\n', i, datasets{i});
    end
    fprintf('\n');
    
    % Initialize storage
    prec_by_dataset = struct();
    
    % Loop through each dataset
    for i_dataset = 1:numel(datasets)
        curr_dataset = datasets{i_dataset};
        fprintf('Processing dataset %d/%d: %s\n', i_dataset, numel(datasets), curr_dataset);
        
        % Get subjects from this dataset
        idx_dataset = find(cellfun(@(d) strcmp(d.dataset, curr_dataset), {fits(1).subj.data}));
        
        if isempty(idx_dataset)
            fprintf('  ⚠️  No subjects found for %s, skipping\n\n', curr_dataset);
            continue;
        end
        
        fprintf('  Found %d subjects\n', numel(idx_dataset));
        
        % Generate simulations and fit for each k level
        [sims_dataset, params_dataset] = deal([]);
        
        for k = 0:3
            fprintf('    Simulating k=%d... ', k);
            
            % Simulate with parameter override
            [new_sims, new_params] = BAKR_2024_simulate_data(fits(1), idx_dataset, k);
            
            % Accumulate
            params_dataset = [params_dataset; new_params];
            sims_dataset = [sims_dataset new_sims];
            
            fprintf('done (%d sims)\n', numel(new_sims));
        end
        
        fprintf('  Fitting %d simulations... ', numel(sims_dataset));
        
        % Fit the simulated data
        model = fits(1).model;
        fits_est = mn_fit(sims_dataset', model);
        
        fprintf('done\n');
        
        % Extract estimates
        estimates = zeros(numel(fits_est.subj), numel(model.params));
        for i_subj = 1:numel(fits_est.subj)
            param_struct = fits_est.subj(i_subj).params;
            for p = 1:numel(model.params)
                estimates(i_subj, p) = param_struct.(model.params(p).name);
            end
        end
        
        % Store results
        prec_by_dataset.(strrep(curr_dataset, '-', '_')).model = model;
        prec_by_dataset.(strrep(curr_dataset, '-', '_')).params.est = estimates;
        prec_by_dataset.(strrep(curr_dataset, '-', '_')).params.gen = params_dataset;
        prec_by_dataset.(strrep(curr_dataset, '-', '_')).dataset = curr_dataset;
        prec_by_dataset.(strrep(curr_dataset, '-', '_')).sims = sims_dataset;
        
        fprintf('  ✓ Parameter recovery complete for %s\n\n', curr_dataset);
    end
    
    % Save results
    save(fullfile(output_dir,'supplementary','parameter_recovery_by_dataset.mat'), 'prec_by_dataset');
    
    fprintf('✓ All datasets complete\n');
    fprintf('  Saved: parameter_recovery_by_dataset.mat\n\n');
    
catch e
    warning('Parameter recovery by dataset failed: %s', e.message);
    fprintf('  Stack trace:\n');
    for i = 1:length(e.stack)
        fprintf('    [%d] %s (line %d)\n', i, e.stack(i).name, e.stack(i).line);
    end
    fprintf('\n');
end

fprintf('========================================\n\n');

%% effect of lowering the upper bound on gamma
try
    fprintf('Testing effect of gamma bounds...\n');
    
    % Reload and preprocess data
    load(fullfile(project_folder,'data','llm_data.mat'));
    if ismember('n_trials', data.Properties.VariableNames)
        data.n_trials = [];
    end
    data.n_blocks(:) = 1;
    data = mn_table2struct(data,'subjID','remove_redundancy',...
                          'exceptions',{'choice_own','choice_other','missing'});
    
    model = BAKR_2024_CHASE_config('CH','fitted',3,'RW-freq');
    gamma = arrayfun(@(subj) subj.params.gamma, fits(1).subj(idx_valid_fits));
    idx_data = idx_valid_fits(gamma > 2.5);
    
    if ~isempty(idx_data)
        UBs = [Inf,10,5,2.5,1,0.5];
        
        for i_UB = 1:numel(UBs)
            UB = UBs(i_UB);
            models{i_UB} = model;
            models{i_UB}.params(3).support = [0,UB];
            if UB == Inf, models{i_UB}.params(3).space = 'log'; end
            models{i_UB}.params(3).grid = linspace(1e-4,min(2,UB-0.01),5);
            models{i_UB}.name = ['CHASE_gamma_max_' num2str(UB)];
        end
        
        fits_bounded = mn_fit(data(idx_data), models);
        save(fullfile(output_dir,'supplementary','fits_bounded.mat'),'fits_bounded');
        
        fprintf('✓ Gamma bounds analysis complete\n\n');
    else
        fprintf('⊘ Skipping gamma bounds (no subjects with gamma > 2.5)\n\n');
    end
catch e
    warning('Gamma bounds analysis failed: %s', e.message);
end

%% level estimation in human-human data
fprintf('⊘ Skipping human-human level estimation (not applicable to LLM data)\n\n');

fprintf('========================================\n');
fprintf('SUPPLEMENTARY ANALYSES COMPLETE\n');
fprintf('========================================\n\n');