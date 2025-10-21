%% CREATE_LLM_SUBSET.m - Extract one subject per dataset

clear; clc;

project_folder = cd;

fprintf('========================================\n');
fprintf('CREATING LLM SUBSET DATA\n');
fprintf('========================================\n\n');

%% Load full LLM data
input_file = fullfile(project_folder, 'data', 'llm_data.mat');
output_file = fullfile(project_folder, 'data', 'llm_data_1.mat');

if ~exist(input_file, 'file')
    error('Input file not found: %s', input_file);
end

fprintf('Loading: %s\n', input_file);
load(input_file, 'data');

fprintf('Full data: %d rows, %d subjects\n\n', height(data), numel(unique(data.subjID)));

%% Get unique datasets
datasets = unique(data.dataset);
fprintf('Found %d unique datasets:\n', numel(datasets));
for i = 1:numel(datasets)
    fprintf('  %d. %s\n', i, datasets{i});
end
fprintf('\n');

%% Extract one subject from each dataset
data_subset = [];

for i = 1:numel(datasets)
    dataset_name = datasets{i};
    
    % Get subjects in this dataset
    idx_dataset = strcmp(data.dataset, dataset_name);
    subjects_in_dataset = unique(data.subjID(idx_dataset));
    
    % Take first subject
    selected_subject = subjects_in_dataset(1);
    
    % Extract all rows for this subject
    idx_subject = (data.subjID == selected_subject);
    subject_data = data(idx_subject, :);
    
    % Append to subset
    data_subset = [data_subset; subject_data];
    
    fprintf('Dataset: %-20s | Selected subject: %d | Rows: %d\n', ...
        dataset_name, selected_subject, sum(idx_subject));
end

%% Save subset
data = data_subset;

fprintf('\n========================================\n');
fprintf('Subset created:\n');
fprintf('  Total rows: %d\n', height(data));
fprintf('  Total subjects: %d\n', numel(unique(data.subjID)));
fprintf('  Datasets: %s\n', strjoin(unique(data.dataset), ', '));
fprintf('========================================\n\n');

fprintf('Saving to: %s\n', output_file);
save(output_file, 'data');

fprintf('\n✅ SUBSET CREATED SUCCESSFULLY!\n\n');

%% Verify the subset
fprintf('========================================\n');
fprintf('VERIFICATION\n');
fprintf('========================================\n\n');

load(output_file, 'data');

subjects = unique(data.subjID);
for i = 1:numel(subjects)
    subj = subjects(i);
    idx = (data.subjID == subj);
    
    dataset_name = data.dataset{find(idx, 1)};
    n_trials = sum(idx);
    
    if ismember('block', data.Properties.VariableNames)
        n_blocks = numel(unique(data.block(idx)));
    else
        n_blocks = 'N/A';
    end
    
    fprintf('Subject %d: Dataset=%-20s | Trials=%d | Blocks=%s\n', ...
        subj, dataset_name, n_trials, num2str(n_blocks));
end

fprintf('\n========================================\n');