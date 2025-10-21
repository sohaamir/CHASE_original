%% Convert LLM CSV data to MATLAB format with proper block/trial structure
%
% This script:
% 1. Loads CSV data with mixed structures (LLM vs Human)
% 2. **RESTRUCTURES LLM data: combines single-bot runs into multi-block subjects**
% 3. Fixes block and trial numbering
% 4. Handles missing data properly
% 5. Removes duplicates
% 6. Validates MERLIN compatibility

%% Set paths

clc; clearvars;

project_folder = '/Users/aamirsohail/University of Birmingham - ALPN Laboratory - Research - Projects/online_social_study/otree_projects/tomLLM_analysis/';
csv_path = fullfile(project_folder, 'rps', 'data', 'processed', 'models', 'llm_data.csv');

% ADD MERLIN TOOLBOX TO PATH
osf_folder = cd;
addpath(fullfile(osf_folder, 'source'));
addpath(fullfile(osf_folder, 'source', 'MERLIN_toolbox'));
fprintf('Added MERLIN toolbox to path from: %s\n', osf_folder);

% Check if CSV exists
if ~exist(csv_path, 'file')
    error('CSV file not found: %s', csv_path);
end

%% Load CSV data

fprintf('\n=== LOADING LLM DATA ===\n');
fprintf('Loading CSV from: %s\n', csv_path);

data_raw = readtable(csv_path);

fprintf('Loaded: %d rows, %d columns\n', height(data_raw), width(data_raw));
fprintf('Subjects: %d\n', length(unique(data_raw.subjID)));
fprintf('Columns: %s\n', strjoin(data_raw.Properties.VariableNames, ', '));

%% Convert data types

fprintf('\n=== CONVERTING DATA TYPES ===\n');

% Ensure all numeric fields are double
numeric_fields = {'subjID', 'trial', 'bot_level', 'choice_own', 'choice_other', ...
                 'score_own', 'score_other', 'missing', 'n_blocks', 'n_trials'};

for i = 1:length(numeric_fields)
    field = numeric_fields{i};
    if ismember(field, data_raw.Properties.VariableNames)
        data_raw.(field) = double(data_raw.(field));
    end
end

fprintf('Converted numeric fields to double\n');

% Ensure dataset is a cell array of strings
if ismember('dataset', data_raw.Properties.VariableNames)
    if ~iscell(data_raw.dataset)
        data_raw.dataset = cellstr(data_raw.dataset);
    end
    fprintf('Dataset field: %s\n', strjoin(unique(data_raw.dataset), ', '));
else
    fprintf('⚠ WARNING: No dataset field found - will create one\n');
    data_raw.dataset = repmat({'UNKNOWN'}, height(data_raw), 1);
end

%% Create initial block field if missing

fprintf('\n=== CREATING BLOCK FIELD ===\n');

if ~ismember('block', data_raw.Properties.VariableNames)
    fprintf('Block field not found - creating from trial numbers...\n');
    
    % Initialize block column
    data_raw.block = zeros(height(data_raw), 1);
    
    subjects = unique(data_raw.subjID);
    for i = 1:length(subjects)
        subj_id = subjects(i);
        idx = (data_raw.subjID == subj_id);
        n_trials = sum(idx);
        
        % Infer blocks from trial count
        if n_trials == 120
            % 3 blocks of 40 trials (HUMAN or multi-block LLM)
            % Assign based on row position (assuming CSV is ordered)
            subj_rows = find(idx);
            data_raw.block(subj_rows(1:40)) = 1;
            data_raw.block(subj_rows(41:80)) = 2;
            data_raw.block(subj_rows(81:120)) = 3;
        elseif n_trials == 40
            % Single block (LLM playing one bot level)
            data_raw.block(idx) = 1;
        else
            % Other trial counts - create blocks of 40
            subj_rows = find(idx);
            for t = 1:n_trials
                block_num = floor((t-1)/40) + 1;
                data_raw.block(subj_rows(t)) = block_num;
            end
            fprintf('  Subject %d: %d trials → %d blocks\n', ...
                    subj_id, n_trials, max(data_raw.block(idx)));
        end
    end
    
    fprintf('✓ Block field created\n');
else
    fprintf('Block field already exists\n');
end

%% DIAGNOSTIC: Check current structure

fprintf('\n=== DIAGNOSTIC: CURRENT DATA STRUCTURE ===\n');

subjects = unique(data_raw.subjID);
for i = 1:min(5, length(subjects))  % Check first 5 subjects
    subj_id = subjects(i);
    subj_data = data_raw(data_raw.subjID == subj_id, :);
    dataset_name = subj_data.dataset{1};
    
    fprintf('\nSubject %d (%s):\n', subj_id, dataset_name);
    fprintf('  Total rows: %d\n', height(subj_data));
    fprintf('  Unique blocks: %s\n', mat2str(unique(subj_data.block)'));
    fprintf('  Unique bot_levels: %s\n', mat2str(unique(subj_data.bot_level)'));
end
fprintf('========================================\n');

%% Fix missing data

fprintf('\n=== FIXING MISSING DATA ===\n');

% Mark NaN choices as missing
missing_own = isnan(data_raw.choice_own);
missing_other = isnan(data_raw.choice_other);
missing_idx = missing_own | missing_other;

if sum(missing_idx) > 0
    fprintf('Found %d trials with NaN choices\n', sum(missing_idx));
    data_raw.missing(missing_idx) = 1;
    data_raw.choice_own(missing_idx) = 1;
    data_raw.choice_other(missing_idx) = 1;
    fprintf('Marked as missing and set valid placeholder choices (1)\n');
else
    fprintf('No missing choices found\n');
end

%% Add required game parameters

fprintf('\n=== ADDING GAME PARAMETERS ===\n');

n_rows = height(data_raw);

if ~ismember('strat_space', data_raw.Properties.VariableNames)
    data_raw.strat_space = repmat(3, n_rows, 1);
end
if ~ismember('win', data_raw.Properties.VariableNames)
    data_raw.win = repmat(1, n_rows, 1);
end
if ~ismember('loss', data_raw.Properties.VariableNames)
    data_raw.loss = repmat(-1, n_rows, 1);
end
if ~ismember('tie', data_raw.Properties.VariableNames)
    data_raw.tie = repmat(0, n_rows, 1);
end

fprintf('Game parameters ready (strat_space=3, win=1, loss=-1, tie=0)\n');

%% ============================================================================
%% CRITICAL: RESTRUCTURE LLM SUBJECTS TO MULTI-BLOCK FORMAT (OPTION A)
%% ============================================================================

fprintf('\n=== RESTRUCTURING LLM SUBJECTS (OPTION A: MULTI-BLOCK) ===\n');
fprintf('Creating multi-block subjects from bot-level runs...\n\n');

% Process each LLM dataset
llm_datasets = {'DEEPSEEK-NORMAL', 'DEEPSEEK-SCOT', 'GPT-NORMAL'};
data_restructured = table();

% Start with a new subject ID for restructured LLMs
new_subj_id = 1000;

for d = 1:length(llm_datasets)
    dataset_name = llm_datasets{d};
    
    % Find all subjects in this dataset
    idx_dataset = strcmp(data_raw.dataset, dataset_name);
    subjects_in_dataset = unique(data_raw.subjID(idx_dataset));
    
    if isempty(subjects_in_dataset)
        fprintf('  No subjects found for %s - skipping\n', dataset_name);
        continue;
    end
    
    fprintf('Processing %s:\n', dataset_name);
    fprintf('  Found %d subjects in raw data\n', length(subjects_in_dataset));
    
    % Organize subjects by bot level
    subj_by_bot = struct('k0', [], 'k1', [], 'k2', [], 'multi', []);
    
    for s = 1:length(subjects_in_dataset)
        subj_id = subjects_in_dataset(s);
        subj_data = data_raw(data_raw.subjID == subj_id & idx_dataset, :);
        
        % Determine structure
        n_trials = height(subj_data);
        unique_bots = unique(subj_data.bot_level);
        unique_blocks = unique(subj_data.block);
        
        if length(unique_bots) == 1 && n_trials == 40
            % Single bot level, 40 trials - needs to be combined
            bot_level = unique_bots(1);
            
            switch bot_level
                case 0
                    subj_by_bot.k0 = [subj_by_bot.k0; subj_id];
                case 1
                    subj_by_bot.k1 = [subj_by_bot.k1; subj_id];
                case 2
                    subj_by_bot.k2 = [subj_by_bot.k2; subj_id];
            end
            
        elseif length(unique_blocks) == 3 && n_trials == 120
            % Already multi-block - keep as is
            subj_by_bot.multi = [subj_by_bot.multi; subj_id];
            fprintf('    Subject %d: already multi-block (keeping as-is)\n', subj_id);
            
            % Add to restructured data with updated ID
            subj_data.subjID(:) = new_subj_id;
            data_restructured = [data_restructured; subj_data];
            new_subj_id = new_subj_id + 1;
            
        else
            fprintf('    ⚠ Subject %d: unexpected structure (trials=%d, blocks=%d, bots=%s)\n', ...
                    subj_id, n_trials, length(unique_blocks), mat2str(unique_bots'));
        end
    end
    
    % Report what we found
    fprintf('  Single-bot subjects: k=0:%d, k=1:%d, k=2:%d\n', ...
            length(subj_by_bot.k0), length(subj_by_bot.k1), length(subj_by_bot.k2));
    
    % Determine how many complete triplets we can form
    n_complete_triplets = min([length(subj_by_bot.k0), ...
                               length(subj_by_bot.k1), ...
                               length(subj_by_bot.k2)]);
    
    if n_complete_triplets == 0
        fprintf('  ⚠ Cannot form any complete triplets (need subjects at all 3 bot levels)\n\n');
        continue;
    end
    
    fprintf('  → Creating %d multi-block subjects (triplets)\n', n_complete_triplets);
    
    % Create multi-block subjects from triplets
    for triplet_idx = 1:n_complete_triplets
        
        combined_data = table();
        
        % Add k=0 block
        subj_k0 = subj_by_bot.k0(triplet_idx);
        block_data_k0 = data_raw(data_raw.subjID == subj_k0 & idx_dataset, :);
        block_data_k0.subjID(:) = new_subj_id;
        block_data_k0.block(:) = 1;
        block_data_k0.trial = (1:height(block_data_k0))';
        block_data_k0.n_blocks(:) = 3;
        combined_data = [combined_data; block_data_k0];
        
        % Add k=1 block
        subj_k1 = subj_by_bot.k1(triplet_idx);
        block_data_k1 = data_raw(data_raw.subjID == subj_k1 & idx_dataset, :);
        block_data_k1.subjID(:) = new_subj_id;
        block_data_k1.block(:) = 2;
        block_data_k1.trial = (1:height(block_data_k1))';
        block_data_k1.n_blocks(:) = 3;
        combined_data = [combined_data; block_data_k1];
        
        % Add k=2 block
        subj_k2 = subj_by_bot.k2(triplet_idx);
        block_data_k2 = data_raw(data_raw.subjID == subj_k2 & idx_dataset, :);
        block_data_k2.subjID(:) = new_subj_id;
        block_data_k2.block(:) = 3;
        block_data_k2.trial = (1:height(block_data_k2))';
        block_data_k2.n_blocks(:) = 3;
        combined_data = [combined_data; block_data_k2];
        
        % Verify
        if height(combined_data) == 120
            data_restructured = [data_restructured; combined_data];
            
            if triplet_idx <= 3  % Show first 3
                fprintf('    Subject %d: combined (%d,%d,%d) → 120 trials\n', ...
                        new_subj_id, subj_k0, subj_k1, subj_k2);
            end
            
            new_subj_id = new_subj_id + 1;
        else
            fprintf('    ⚠ Triplet %d: incorrect total (%d trials)\n', ...
                    triplet_idx, height(combined_data));
        end
    end
    
    % Report any unused subjects
    n_unused_k0 = length(subj_by_bot.k0) - n_complete_triplets;
    n_unused_k1 = length(subj_by_bot.k1) - n_complete_triplets;
    n_unused_k2 = length(subj_by_bot.k2) - n_complete_triplets;
    
    if n_unused_k0 > 0 || n_unused_k1 > 0 || n_unused_k2 > 0
        fprintf('  ⚠ Unused subjects (incomplete triplets): k=0:%d, k=1:%d, k=2:%d\n', ...
                n_unused_k0, n_unused_k1, n_unused_k2);
    end
    
    fprintf('  ✓ Created %d multi-block subjects for %s\n\n', n_complete_triplets, dataset_name);
end

%% Process HUMAN subjects (keep multi-block structure)

fprintf('=== PROCESSING HUMAN SUBJECTS ===\n');

human_subjects = unique(data_raw.subjID(strcmp(data_raw.dataset, 'HUMAN')));

for s = 1:length(human_subjects)
    subj_id = human_subjects(s);
    subj_data = data_raw(data_raw.subjID == subj_id, :);
    
    if s <= 3  % Show first 3
        fprintf('Subject %d (HUMAN): %d trials, %d blocks\n', ...
                subj_id, height(subj_data), length(unique(subj_data.block)));
    end
    
    % Sort by block and trial
    subj_data = sortrows(subj_data, {'block', 'trial'});
    
    % Fix trial numbering within each block
    for block = unique(subj_data.block)'
        idx_block = (subj_data.block == block);
        subj_data.trial(idx_block) = (1:sum(idx_block))';
    end
    
    % Ensure n_blocks is correct
    subj_data.n_blocks(:) = length(unique(subj_data.block));
    
    data_restructured = [data_restructured; subj_data];
end

fprintf('Processed %d HUMAN subjects\n', length(human_subjects));

fprintf('\n✓ Restructuring complete!\n');
fprintf('  Total subjects after restructuring: %d\n', length(unique(data_restructured.subjID)));
fprintf('  Total trials: %d\n', height(data_restructured));

% Replace with restructured data
data_fixed = data_restructured;

%% Add opponent type metadata

fprintf('\n=== ADDING OPPONENT TYPE METADATA ===\n');

if ~ismember('opp_type', data_fixed.Properties.VariableNames)
    data_fixed.opp_type = cell(height(data_fixed), 1);
    for i = 1:height(data_fixed)
        if ~isnan(data_fixed.bot_level(i))
            data_fixed.opp_type{i} = sprintf('bot_level_%d', data_fixed.bot_level(i));
        else
            data_fixed.opp_type{i} = 'bot';
        end
    end
    fprintf('Created opp_type field from bot_level\n');
end

%% Remove duplicates

fprintf('\n=== CHECKING FOR DUPLICATES ===\n');

[~, unique_idx] = unique(data_fixed(:, {'subjID', 'block', 'trial'}), 'rows', 'stable');
n_duplicates = height(data_fixed) - length(unique_idx);

if n_duplicates > 0
    fprintf('⚠ Found %d duplicate (subjID, block, trial) tuples\n', n_duplicates);
    fprintf('  Keeping first occurrence of each\n');
    data_fixed = data_fixed(unique_idx, :);
    fprintf('  After deduplication: %d rows\n', height(data_fixed));
else
    fprintf('✓ No duplicates found\n');
end

%% ============================================================================
%% VALIDATION: Verify Multi-Block Structure
%% ============================================================================

fprintf('\n=== VALIDATION: MULTI-BLOCK STRUCTURE ===\n\n');

subjects = unique(data_fixed.subjID);
validation_passed = true;

for i = 1:length(subjects)
    subj_id = subjects(i);
    subj_data = data_fixed(data_fixed.subjID == subj_id, :);
    dataset_name = subj_data.dataset{1};
    
    fprintf('Subject %d (%s):\n', subj_id, dataset_name);
    fprintf('  Total trials: %d\n', height(subj_data));
    
    % Check blocks
    unique_blocks = unique(subj_data.block);
    fprintf('  Blocks: %s\n', mat2str(unique_blocks'));
    
    % Check bot levels
    unique_bots = unique(subj_data.bot_level);
    fprintf('  Bot levels: %s\n', mat2str(unique_bots'));
    
    % Verify each block
    for block = unique_blocks'
        idx_block = (subj_data.block == block);
        n_trials_block = sum(idx_block);
        trials_in_block = subj_data.trial(idx_block);
        expected_trials = (1:n_trials_block)';
        bots_in_block = unique(subj_data.bot_level(idx_block));
        
        % Check trial numbering
        if ~isequal(trials_in_block, expected_trials)
            fprintf('    ✗ Block %d: INCORRECT trial numbering\n', block);
            validation_passed = false;
        else
            fprintf('    ✓ Block %d: %d trials (1:%d)', block, n_trials_block, n_trials_block);
        end
        
        % Check bot level homogeneity within block
        if length(bots_in_block) == 1
            fprintf(', bot_level=%d ✓\n', bots_in_block);
        else
            fprintf(', MIXED bot_levels=%s ✗\n', mat2str(bots_in_block'));
            validation_passed = false;
        end
    end
    
    % Expected structure check
    if contains(dataset_name, {'DEEPSEEK', 'GPT', 'LLM'})
        % LLMs should have 3 blocks of 40 trials each
        if length(unique_blocks) ~= 3
            fprintf('  ✗ Expected 3 blocks for LLM, got %d\n', length(unique_blocks));
            validation_passed = false;
        end
        if height(subj_data) ~= 120
            fprintf('  ✗ Expected 120 trials for LLM, got %d\n', height(subj_data));
            validation_passed = false;
        end
        % Should have one block per bot level
        if ~isequal(sort(unique_bots), [0;1;2])
            fprintf('  ✗ Expected bot_levels [0,1,2], got %s\n', mat2str(unique_bots'));
            validation_passed = false;
        end
    elseif strcmp(dataset_name, 'HUMAN')
        % Humans should have 3 blocks of 40 trials each
        if length(unique_blocks) ~= 3
            fprintf('  ⚠ Expected 3 blocks for HUMAN, got %d\n', length(unique_blocks));
        end
        if height(subj_data) ~= 120
            fprintf('  ⚠ Expected 120 trials for HUMAN, got %d\n', height(subj_data));
        end
    end
    
    fprintf('\n');
end

if validation_passed
    fprintf('✓ ALL VALIDATION CHECKS PASSED\n\n');
else
    warning('Some validation checks failed - review output above');
end

%% VALIDATE MERLIN compatibility

fprintf('\n=== VALIDATING MERLIN COMPATIBILITY ===\n');

fprintf('Testing mn_table2struct conversion...\n');
try
    test_struct = mn_table2struct(data_fixed, 'subjID', 'remove_redundancy', ...
                                 'exceptions', {'choice_own', 'choice_other', 'missing'}, ...
                                 'block_var', 'block');
    
    fprintf('  ✓ SUCCESS: Converted to %d subjects\n', length(test_struct));
    
    % Verify structure for each subject
    fprintf('\nVerifying struct format:\n');
    for i = 1:length(test_struct)
        subj = test_struct(i);
        dataset_name = subj.dataset{1};
        
        fprintf('  Subject %d (%s):\n', subj.subjID, dataset_name);
        fprintf('    n_blocks: %d\n', subj.n_blocks);
        fprintf('    n_trials: %s\n', mat2str(subj.n_trials));
        fprintf('    choice_own size: %s\n', mat2str(size(subj.choice_own)));
        
        % Verify vector fields have correct dimensions
        if size(subj.choice_own, 2) ~= subj.n_blocks
            fprintf('    ✗ choice_own has %d columns, expected %d\n', ...
                    size(subj.choice_own, 2), subj.n_blocks);
            validation_passed = false;
        else
            fprintf('    ✓ choice_own: [40 × %d] format correct\n', subj.n_blocks);
        end
        
        % Check bot_level structure
        if isfield(subj, 'bot_level')
            fprintf('    bot_level size: %s\n', mat2str(size(subj.bot_level)));
            if size(subj.bot_level, 2) == subj.n_blocks
                unique_per_block = arrayfun(@(b) unique(subj.bot_level(:,b)), ...
                                           1:subj.n_blocks);
                fprintf('    bot_levels per block: %s ✓\n', mat2str(unique_per_block));
            end
        end
    end
    
catch ME
    fprintf('  ✗ ERROR: Struct conversion failed\n');
    fprintf('    Message: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('    Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
    validation_passed = false;
    error('MERLIN compatibility validation failed');
end

%% Save data

fprintf('\n=== SAVING DATA ===\n');

output_path = fullfile(osf_folder, 'data', 'llm_data.mat');

osf_data_dir = fileparts(output_path);
if ~exist(osf_data_dir, 'dir')
    mkdir(osf_data_dir);
    fprintf('Created directory: %s\n', osf_data_dir);
end

% Save both table and struct formats
data = data_fixed;
data_struct = test_struct;

save(output_path, 'data', 'data_struct');

fprintf('✓ Data saved to: %s\n', output_path);
fprintf('  Format: Multi-block (Option A)\n');

%% Final summary

fprintf('\n=== CONVERSION COMPLETE ===\n');
fprintf('Final structure (OPTION A - Multi-Block):\n');
fprintf('  Total subjects: %d\n', length(unique(data.subjID)));
fprintf('  Total trials: %d\n', height(data));
fprintf('  Missing trials: %d (%.1f%%)\n', sum(data.missing), sum(data.missing)/height(data)*100);

fprintf('\nBy dataset:\n');
datasets = unique(data.dataset);
for d = 1:length(datasets)
    dataset_name = datasets{d};
    idx = strcmp(data.dataset, dataset_name);
    subj_in_dataset = unique(data.subjID(idx));
    
    fprintf('  %s:\n', dataset_name);
    fprintf('    - Subjects: %d\n', length(subj_in_dataset));
    fprintf('    - Trials: %d (%.0f per subject)\n', sum(idx), sum(idx)/length(subj_in_dataset));
    
    % Check a sample subject
    if ~isempty(subj_in_dataset)
        sample_subj = subj_in_dataset(1);
        sample_data = data(data.subjID == sample_subj, :);
        n_blocks = length(unique(sample_data.block));
        bot_levels = unique(sample_data.bot_level);
        
        fprintf('    - Structure: %d blocks, bot_levels=%s\n', n_blocks, mat2str(bot_levels'));
    end
end

fprintf('\nOption A Implementation:\n');
fprintf('  ✓ Each LLM model = 1 subject with 3 blocks\n');
fprintf('  ✓ Each block = 40 trials vs one bot level (k=0,1,2)\n');
fprintf('  ✓ Parameters will reset at each block boundary\n');
fprintf('  ✓ Comparable to HUMAN subjects (also 3 blocks)\n');

if validation_passed
    fprintf('\n✓✓✓ ALL VALIDATION CHECKS PASSED ✓✓✓\n');
    fprintf('Data is ready for BAKR_2024_run_model_fitting.m\n');
else
    warning('Some validation issues detected - please review');
end