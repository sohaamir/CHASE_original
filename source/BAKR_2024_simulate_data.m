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
            
            % === FIX: Safely extract bot levels ===
            % Problem: Original code used bot_level(1:40:end) which can
            % exceed array bounds when end/40 is not a whole number
            bot_level_full = subj_data.bot_level;
            n_trials_total = length(bot_level_full);
            n_blocks_actual = floor(n_trials_total / 40);
            
            % Sample one bot level per block (every 40 trials)
            % Ensure we don't exceed array bounds
            safe_indices = 1:40:(40 * n_blocks_actual);
            
            % Validate we're not accessing beyond array
            assert(max(safe_indices) <= n_trials_total, ...
                'Index exceeds bot_level array length');
            
            task.bot.levels = bot_level_full(safe_indices);
            % ======================================
            
            % Update task structure to match actual data
            task.n_trials = 40;  % trials per block
            task.n_blocks = n_blocks_actual;
            
            % Extract parameters
            curr_params = cell2mat(struct2cell(fits(i_model).subj(idx(i_subj)).params))';
            
            % Allow for overriding of level parameter (for parameter recovery)
            if nargin > 2
                curr_params(end) = k;
                params(i_subj, :) = curr_params;
                subjID = k * 100 + i_subj;
            else
                subjID = i_model * 100 + i_subj;
            end
            
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