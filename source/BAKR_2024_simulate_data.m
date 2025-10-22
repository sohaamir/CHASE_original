%% BAKR_2024_simulate_data.m (FIXED VERSION)
% Save this as: source/BAKR_2024_simulate_data.m
% This replaces the existing file with proper bounds checking

function [sims, params] = BAKR_2024_simulate_data(fits, idx, k)
% BAKR_2024_simulate_data - Simulate data from fitted models
% 
% Fixed version that properly handles variable-length bot_level arrays
% Key fix: Uses safe indexing (1:40:40*n_blocks) instead of (1:40:end)

    task = mn_RPS_config;
    
    % Initialize outputs
    if nargin < 3
        params = [];
    end
    
    for i_model = 1:numel(fits)
        
        curr_model = fits(i_model).model;
        curr_model.sim_fxn = curr_model.loglik_fxn;
        
        for i_subj = 1:numel(idx)
            
            % Extract subject-specific info
            subj_data = fits(1).subj(idx(i_subj)).data;
            
            % === DETAILED DIAGNOSTICS ===
            if i_model == 2 && i_subj == 4
                fprintf('\n=== DETAILED DEBUG: Model %d, Subj %d ===\n', i_model, i_subj);
                fprintf('subj_data fields: %s\n', strjoin(fieldnames(subj_data), ', '));
                fprintf('subj_data.subjID: %d\n', subj_data.subjID);
                
                if isfield(subj_data, 'bot_level')
                    fprintf('bot_level exists: YES\n');
                    fprintf('  Type: %s\n', class(subj_data.bot_level));
                    fprintf('  Size: %s\n', mat2str(size(subj_data.bot_level)));
                    fprintf('  Length: %d\n', length(subj_data.bot_level));
                    fprintf('  First 10 values: %s\n', mat2str(subj_data.bot_level(1:min(10,end))));
                    if length(subj_data.bot_level) <= 20
                        fprintf('  ALL values: %s\n', mat2str(subj_data.bot_level));
                    end
                else
                    fprintf('bot_level exists: NO\n');
                end
                
                if isfield(subj_data, 'n_trials')
                    fprintf('n_trials: %d\n', subj_data.n_trials);
                end
                if isfield(subj_data, 'n_blocks')
                    fprintf('n_blocks: %d\n', subj_data.n_blocks);
                end
                
                fprintf('=== END DEBUG ===\n\n');
            end
            
            % === FIX: Safely extract bot levels ===
            bot_level_full = subj_data.bot_level;
            n_trials_total = length(bot_level_full);
            
            fprintf('[Model %d, Subj %d] bot_level length: %d\n', i_model, i_subj, n_trials_total);
            
            % Check if bot_level is already in compact format
            if n_trials_total <= 10
                fprintf('[Model %d, Subj %d] Using compact format\n', i_model, i_subj);
                task.bot.levels = bot_level_full;
                n_blocks_actual = length(bot_level_full);
            else
                fprintf('[Model %d, Subj %d] Using long format, extracting indices\n', i_model, i_subj);
                n_blocks_actual = floor(n_trials_total / 40);
                safe_indices = 1:40:(40 * n_blocks_actual);
                
                fprintf('[Model %d, Subj %d] n_blocks_actual: %d, safe_indices: %s\n', ...
                    i_model, i_subj, n_blocks_actual, mat2str(safe_indices));
                
                % THIS IS LINE 51 - where the error occurs
                assert(max(safe_indices) <= n_trials_total, ...
                    'Index exceeds bot_level array length: max_index=%d, array_length=%d', ...
                    max(safe_indices), n_trials_total);
                
                task.bot.levels = bot_level_full(safe_indices);
            end
            
            fprintf('[Model %d, Subj %d] Final task.bot.levels: %s\n', ...
                i_model, i_subj, mat2str(task.bot.levels));
            % ======================================
            
            % Update task structure to match actual data
            task.n_trials = 40;  % trials per block
            task.n_blocks = n_blocks_actual;
            
            % Extract parameters in the CORRECT order matching model definition
            param_struct = fits(i_model).subj(idx(i_subj)).params;
            curr_params = zeros(1, length(curr_model.params));
            for p = 1:length(curr_model.params)
                param_name = curr_model.params(p).name;
                curr_params(p) = param_struct.(param_name);
            end
            
            % Allow for overriding of level parameter (for parameter recovery)
            if nargin > 2
                curr_params(end) = k;
                params(i_subj, :) = curr_params;
                subjID = k * 100 + i_subj;
            else
                subjID = i_model * 100 + i_subj;
            end

            % Add after line 36
            fprintf('DEBUG Model %d, Subj %d:\n', i_model, i_subj);
            fprintf('  Param fields: %s\n', strjoin(fieldnames(fits(i_model).subj(idx(i_subj)).params), ', '));
            fprintf('  Model expects: %s\n', strjoin({curr_model.params.name}, ', '));
            fprintf('  Extracted params length: %d\n', length(curr_params));
            fprintf('  Model params length: %d\n', length(curr_model.params));
            
            % Simulate
            sim = mn_sim(task, curr_model, curr_params);
            sim.subj.data.subjID = subjID;
            sim.subj.data.model = fits(i_model).model.name;
            sim.subj.data.loss = -1;
            
            % Add to output
            if i_model == 1 && i_subj == 1
                sims(1) = sim.subj.data;
            else
                sims(end+1) = sim.subj.data;
            end
            
        end
        
    end
    
end