function [sims,params] = BAKR_2024_simulate_data_LLM(fits,idx,k)
%
% Modified version for LLM data with multi-block subjects
%

task = mn_RPS_config;

for i_model = 1:numel(fits)

    curr_model = fits(i_model).model;
    curr_model.sim_fxn = curr_model.loglik_fxn;

    for i_subj = 1:numel(idx)

        % === FIX 1: Handle multi-block parameters ===
        subj_data = fits(i_model).subj(idx(i_subj));
        
        % Check if multi-block fitting was used
        if isstruct(subj_data.params) && numel(subj_data.params) > 1
            % Average parameters across blocks for simulation
            param_names = fieldnames(subj_data.params(1));
            curr_params = zeros(1, numel(param_names));
            for p = 1:numel(param_names)
                param_values = arrayfun(@(b) subj_data.params(b).(param_names{p}), ...
                                       1:numel(subj_data.params));
                curr_params(p) = mean(param_values);
            end
            curr_params = curr_params';  % Make column vector
        else
            % Single block: manual extraction (struct2array doesn't exist)
            param_names = fieldnames(subj_data.params);
            curr_params = zeros(numel(param_names), 1);
            for p = 1:numel(param_names)
                curr_params(p) = subj_data.params.(param_names{p});
            end
        end
        
        % bot_level is already organized per-block after struct conversion
        bot_levels = subj_data.data.bot_level;
        
        % Check the dimensionality to determine format
        if size(bot_levels, 2) == 1 && size(bot_levels, 1) > 3
            % Trial-by-trial format: extract first occurrence of each block
            % Assuming blocks change every 40 trials
            task.bot.levels = bot_levels(1:40:end);
        elseif size(bot_levels, 2) > 1
            % Block format: [40 trials × n_blocks] - extract unique per block
            task.bot.levels = bot_levels(1, :);  % First row contains the level for each block
        elseif numel(bot_levels) == 3
            % Already in compact block format [0 1 2]
            task.bot.levels = bot_levels;
        else
            % Fallback: try to extract unique values
            if numel(unique(bot_levels)) == 3
                task.bot.levels = unique(bot_levels)';
            else
                error('Cannot determine bot_level structure for subject %d', subj_data.subjID);
            end
        end
        
        % === FIX 3: Only override k parameter if provided ===
        if nargin > 2  % Changed from > 1 to > 2
            curr_params(end) = k;
            params(i_subj,:) = curr_params;
            subjID = k*100 + i_subj;
        else
            subjID = i_model*100 + i_subj;
            if nargout > 1
                params(i_subj,:) = curr_params;
            end
        end

        % simulate
        sim = mn_sim(task,curr_model,curr_params);
        sim.subj.data.subjID = subjID;
        sim.subj.data.model = fits(i_model).model.name;
        sim.subj.data.loss = -1;

        % add
        if i_model == 1 && i_subj == 1
            sims(1) = sim.subj.data;
        else
            sims(end+1) = sim.subj.data;
        end

    end

end

end