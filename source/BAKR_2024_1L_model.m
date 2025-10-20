function BAKR_2024_1L_model(config,subj)

MODEL_MISSINGS = 1; % include missing trials as additional regressors (choice + missed)
MODEL_ACTION_IDENTITY = 1; % subj action during choice, opp action during feedback

% settings for spm
hpf_threshold = 128;
smoothing_fwhm = 6;
brain_mask = {''};

% unpack
folders = config.folders;
dataset = config.dataset;
model_type = config.type;
pmods = config.pmods;

% create output folder
output_folder = fullfile(folders.results,model_type,'1L',subj);
if ~exist(output_folder,'dir')
    mkdir(output_folder);
end

%% get data

% get runs
files_runs = dir(fullfile(folders.prepro,subj,'*','func',sprintf('s%i.%s_*_run-*_bold.nii',smoothing_fwhm,subj)));
run_idx = (regexp([files_runs(:).name], '(?<=_run-)[0-9]', 'match'))'; % get the run number from the file name
run_name = [run_idx {files_runs.name}'];

[~,sortOrder] = sort(run_name(:,1));
run_name = run_name(sortOrder, 2);

% timing info
scaninfo.RepetitionTime = 2.2382;
scaninfo.MaxSlices = 40;

%% get behavioral data

switch dataset
    
    case 'primary'
        
        % timings/RTs/rewards
        behav_folder = fullfile(folders.prepro,subj,'ses-00001','beh');
        load(fullfile(behav_folder,'behavior_and_timings.mat'),'behav');

        % get fits
        load(fullfile(folders.project,'results','fits_CHASE_struct'),'MN');
        idx_fmri = find(arrayfun(@(subj) contains(subj.data.dataset,'fMRI'),MN.subj));
        subjects_fmri = arrayfun(@(subj) subj.data.subjID_original,MN.subj(idx_fmri));

        % recode SUBID to BIDSID
        subj_mapping = readtable(fullfile(folders.prepro,'participants.xlsx'));
        subj_mapping.SUBID = cellfun(@(x) str2double(x(:,end-7:end-3)), subj_mapping.data_id);
        targetID = subj_mapping.SUBID(strcmp(subj,subj_mapping.participant_id));

        % extract correct subj
        idx_subj_fmri = (subjects_fmri == targetID);
        curr_subj = MN.subj(idx_fmri(idx_subj_fmri));
        assert(targetID == curr_subj.data.subjID_original);
        
    case 'replication'
        
        % folder for nuisance regressors
        behav_folder = fullfile(folders.prepro,subj,'ses-00001','beh');

        % get new fits
        load(fullfile(folders.project,'results','replication','replication_model_fits.mat'),'MN');
        subjects_fmri = arrayfun(@(subj) subj.data.fmriID,MN.subj);

        % recode SUBID to BIDSID
        subj_mapping = readtable(fullfile(folders.prepro,'participants.xlsx'));
        subj_mapping.SUBID = cellfun(@(x) str2double(x(:,end-3:end)), subj_mapping.data_id);
        targetID = subj_mapping.SUBID(strcmp(subj,subj_mapping.participant_id));

        % extract correct subj
        idx_subj_fmri = (subjects_fmri == targetID);
        curr_subj = MN.subj(idx_subj_fmri);
        assert(targetID == curr_subj.data.fmriID);
        
end

%% prepare pmods

idx = ~curr_subj.data.missing;
assert(numel(idx) > 1);

% prepare KL-div
if strcmp(model_type,'pmod_ordinal')
    curr_subj.states.subj_KL_div(idx) = log(curr_subj.states.subj_KL_div(idx)+1e-3); % log-transform if decoding (has to happen *before* z-scoring)
end
curr_subj.states.subj_KL_div(idx) = zscore(curr_subj.states.subj_KL_div(idx)); % z-score (as unbounded)

% add block variable
if ~isfield(curr_subj.data,'block')
    n_blocks = numel(curr_subj.data.bot_level);
    n_trials = curr_subj.data.n_trials/n_blocks;
    curr_subj.data.block = repelem(1:n_blocks,1,n_trials)';
end

% extract and rename
for i_run = 1:numel(curr_subj.data.bot_level)%numel(behav)
    
    idx = (curr_subj.data.block == i_run);
    states(i_run).SV_ch          = curr_subj.states.subj_SV(idx);
    states(i_run).subj_KL_div_fb = curr_subj.states.subj_KL_div(idx);
    states(i_run).APE_fb         = curr_subj.states.subj_APE(idx);
    states(i_run).SV_fb          = curr_subj.states.subj_SV(idx);
    states(i_run).reward_fb      = curr_subj.data.score_own(idx);

    states(i_run).trial_fb      = curr_subj.data.trial(idx)/40;
    states(i_run).trial_2_fb    = states(i_run).trial_fb.^2;
    
    switch dataset
        
        case 'primary'
            
            behav(i_run).subj_p_k = curr_subj.states.beliefs(idx,:); % <- used for level-decoding; requires extended file above
    
        case 'replication'
            
            behav(i_run).missing_trial          = curr_subj.data.missing(idx);
            behav(i_run).onset_choicephase(:,1) = curr_subj.data.onsets.choice(:,i_run);
            behav(i_run).onset_choicephase(:,2) = curr_subj.data.RT(idx);
            behav(i_run).onset_feedback(:,1)    = curr_subj.data.onsets.feedback(:,i_run);
            behav(i_run).onset_feedback(:,2)    = 2;
            behav(i_run).onset_response(:,1)    = curr_subj.data.onsets.response(:,i_run);
            behav(i_run).onset_response(:,2)    = 1;    

            behav(i_run).a_subj = curr_subj.data.choice_own(idx);
            behav(i_run).a_opp  = curr_subj.data.choice_other(idx);
            
    end
    
end
    
%% fMRI model specification

fmri_spec.dir = {output_folder};
fmri_spec.timing.units = 'secs';
fmri_spec.timing.RT = scaninfo.RepetitionTime;
fmri_spec.timing.fmri_t = scaninfo.MaxSlices;
fmri_spec.timing.fmri_t0 = round(scaninfo.MaxSlices/2);
    
%% runs
for i_run = 1:numel(run_name)
    
    % scans
    S.scans = cellstr([spm_select('expand',[fullfile(files_runs(i_run).folder, run_name{i_run})])]);
        
    % exclude missing trials (and model them separately)            
    if MODEL_MISSINGS
        idx_missing = behav(i_run).missing_trial;
    else
        idx_missing = zeros(size(behav(i_run).onset_choicephase));
    end
    assert(numel(idx_missing) > 1);

    %% conditions of no interest
    
%     % ITI
%     S.cond(1).name = 'iti';
%     S.cond(1).onset = behav(i_run).onset_fixation(:,1);
%     S.cond(1).duration = behav(i_run).onset_fixation(:,2);
%     S.cond(1).tmod = 0;
%     S.cond(1).pmod = struct('name', {}, 'param', {}, 'poly', {});
%     S.cond(1).orth = 0;

    %% conditions and pmods
    
    switch model_type

        case 'pmods_whole_run' % whole runs, including one or more pmods, one condition for all outcomes
            
            S.cond(1).name = 'ch';
            S.cond(1).onset = behav(i_run).onset_choicephase(~idx_missing,1);
            S.cond(1).duration = behav(i_run).onset_choicephase(~idx_missing,2);
            S.cond(1).pmod = struct('name',{},'param',{},'poly',{});
            S.cond(1).orth = 0;
        
            S.cond(2).name = 'fb';
            S.cond(2).onset = behav(i_run).onset_feedback(~idx_missing,1);
            S.cond(2).duration = behav(i_run).onset_feedback(~idx_missing,2);
            S.cond(2).pmod = struct('name',{},'param',{},'poly',{});
            S.cond(2).orth = 0;
        
            pmod_count = zeros(1,2); % keep track of how many pmods were entered per condition
            if isa(pmods,'char'), pmods = {pmods}; end
            for i_pmod = 1:numel(pmods)
                curr_pmod = pmods{i_pmod};
                curr_cond = strsplit(curr_pmod,'_');
                switch curr_cond{end}
                    case 'ch', i_cond = 1;
                    case 'fb', i_cond = 2;
                    otherwise, error('Have to provide timing.');
                end
                pmod_count(i_cond) = pmod_count(i_cond) + 1; % update condition count
                assert(~any(isnan(states(i_run).(curr_pmod)(~idx_missing))));
                S.cond(i_cond).pmod(pmod_count(i_cond)).name = curr_pmod;
                S.cond(i_cond).pmod(pmod_count(i_cond)).param = states(i_run).(curr_pmod)(~idx_missing) - mean(states(i_run).(curr_pmod)(~idx_missing));
                S.cond(i_cond).pmod(pmod_count(i_cond)).poly = 1;
            end
            
            if MODEL_ACTION_IDENTITY
                S = add_action_pmods(S,behav(i_run),[1,2],~idx_missing);
            end

        case 'pmod_ordinal' % pmod split up into several levels (for support-vector-regression decoding)
            
            curr_data = states(i_run).(pmods) - mean(states(i_run).(pmods)(~idx_missing));
            assert(all(isnan(curr_data(logical(idx_missing)))));
            
            % get bin counts (and thus valid bins)
            n_bins = 5;
            [n_per_bin,edges] = histcounts(curr_data,n_bins);
            bins = find(n_per_bin > 0);
            n_bins = numel(bins);

            i_bin = 1;
            for curr_bin = bins          
                
                assert(n_per_bin(curr_bin) > 0);
                    
                if curr_bin < n_bins
                    idx = (edges(curr_bin) <= curr_data & curr_data < edges(curr_bin+1) & ~idx_missing);    
                else
                    idx = (edges(curr_bin) <= curr_data & curr_data <= edges(curr_bin+1) & ~idx_missing); % cf histcounts
                end

                assert(any(idx));
                
                idx_ch = logical([0; idx(1:end-1)]);
                S.cond(i_bin).name = sprintf('ch_%s_%i',pmods,curr_bin);
                S.cond(i_bin).onset = behav(i_run).onset_choicephase(idx_ch,1); % because effect of *last* outcome on current choice phase
                S.cond(i_bin).duration = behav(i_run).onset_choicephase(idx_ch,2);
                S.cond(i_bin).pmod = struct('name',{},'param',{},'poly',{});

                S.cond(n_bins+i_bin).name = sprintf('fb_%s_%i',pmods,curr_bin);
                S.cond(n_bins+i_bin).onset = behav(i_run).onset_feedback(idx,1);
                S.cond(n_bins+i_bin).duration = behav(i_run).onset_feedback(idx,2);
                S.cond(n_bins+i_bin).pmod = struct('name',{},'param',{},'poly',{});

                if MODEL_ACTION_IDENTITY
                    S = add_action_pmods(S,behav(i_run),[i_bin,n_bins+i_bin],[idx_ch idx]);
                end

                % save control variables
                curr_run.bin(curr_bin).trials = find(idx)/40;
                curr_run.bin(curr_bin).trials_2 = (find(idx)/40).^2;
                curr_run.bin(curr_bin).RT = behav(i_run).onset_choicephase(idx,2);
                curr_run.bin(curr_bin).APE = states(i_run).APE_fb(idx);
                curr_run.bin(curr_bin).reward = states(i_run).reward_fb(idx);

                i_bin = i_bin + 1;
                    
            end

            % save variables for control analyses
            if ~isempty(bins)
                controls.run(i_run) = curr_run;
                if i_run == numel(run_name)
                    controls.subj_fmri = subj;
                    controls.subj_behav = targetID;
                    save(fullfile(behav_folder,'decoding_controls.mat'),'controls');
                end
            end   
            
        case {'levels_whole_run','levels_thresholded'} % assign trials to levels played

            config.thre_var = pmods; % 'none' or 'subj_p_k'
            config.thre_val = 0.5;
            config.thre_n = 5;
            
            [~,behav(i_run).subj_k] = max(behav(i_run).subj_p_k,[],2);
            
            % identify trials above threshold, and subject level(s)
            if ~strcmp(config.thre_var,'none')
                idx_crit = logical(any(behav(i_run).(config.thre_var) >= config.thre_val,2));
                subj_k = unique(behav(i_run).subj_k(idx_crit));
                if numel(subj_k)>1, warning('More than one level.'); end
                n_it = numel(subj_k) + 1; % for each level, plus all leftovers
            else
                idx_crit = logical(ones(size(behav(i_run).a_subj))); % i.e. all trials
                subj_ks = unique(behav(i_run).subj_k);
                [~,idx_max] = max(histcounts(behav(i_run).subj_k,[-0.5; subj_ks+0.5]));
                subj_k = subj_ks(idx_max);
                n_it = 1;
            end
            idx_all = zeros(size(idx_crit));

            % loop over subject levels, and then leftovers (that didn't meet the threshold)
            n_regs = 0;
            for i_it = 1:n_it

                % assign cond name
                if i_it <= numel(subj_k)
                    curr_suffix = sprintf('k%i',subj_k(i_it));
                    idx_curr = (behav(i_run).subj_k == subj_k(i_it) & idx_crit);
                else
                    curr_suffix = 'leftover';
                    idx_curr = ~any(idx_all,2); % new criterion: garbage trials
                end
                [idx_ch,idx_fb] = deal(logical(idx_curr & ~idx_missing));

                if sum(idx_fb) > config.thre_n || (strcmp(curr_suffix,'leftover') && any(idx_fb))

                    % create the conditions
                    S.cond(n_regs+1).name = sprintf('ch_%s',curr_suffix);
                    S.cond(n_regs+1).onset = behav(i_run).onset_choicephase(idx_ch,1); % because effect of *last* outcome on current choice phase
                    S.cond(n_regs+1).duration = behav(i_run).onset_choicephase(idx_ch,2);
                    S.cond(n_regs+1).pmod = struct('name',{},'param',{},'poly',{});
                    S.cond(n_regs+1).orth = 0;

                    S.cond(n_regs+2).name = sprintf('fb_%s',curr_suffix);
                    S.cond(n_regs+2).onset = behav(i_run).onset_feedback(idx_fb,1);
                    S.cond(n_regs+2).duration = behav(i_run).onset_feedback(idx_fb,2);
                    S.cond(n_regs+2).pmod = struct('name',{},'param',{},'poly',{});
                    S.cond(n_regs+2).orth = 0;

                    % account for actions (own actions during choice, and opponent actions during feedback)
                    if MODEL_ACTION_IDENTITY 
                        S = add_action_pmods(S,behav,[n_regs+1,n_regs+2],[idx_ch idx_fb]);
                    end

                    % update count and collect modelled trials (rest will be captured by leftover regressor)
                    n_regs = n_regs + 2;
                    idx_all(:,n_regs/2) = idx_fb; % (we'll ignore the missing first trial for outcome-dependent choice))

                end
            end
            
    end
        
    % add regressors-of-no-interest for missings (if there are any) <- no outcome here
    if any(idx_missing)

        n_cond = size(S.cond,2);
        S.cond(n_cond+1).name = 'missing-ch';
        S.cond(n_cond+1).onset = behav(i_run).onset_choicephase(logical(idx_missing),1);
        S.cond(n_cond+1).duration = behav(i_run).onset_choicephase(logical(idx_missing),2);
        S.cond(n_cond+1).pmod = struct('name',{},'param',{},'poly',{});
        
        S.cond(n_cond+2).name = 'missing-fb';
        S.cond(n_cond+2).onset = behav(i_run).onset_response(logical(idx_missing),1); % <- this is when the fixation cross turns red (no change for onset_feedback anymore)
        S.cond(n_cond+2).duration = behav(i_run).onset_response(logical(idx_missing),2);
        S.cond(n_cond+2).pmod = struct('name',{},'param',{},'poly',{});
        
    end

    % nuisance regressors (motion, global signal, physio)
    included_regressors = 'physio_globalsignal';
    nuisance_file = fullfile(behav_folder,sprintf('nuisance_%s_%s_run_%i.mat',included_regressors,subj,i_run));
    BAKR_2024_create_nuisance_regressors(nuisance_file,folders,subj,i_run);
    
    % non-convolved regressors
    S.multi = {''};
    S.regress = struct('name', {}, 'val', {});
    S.multi_reg = {nuisance_file}; % note there should be always one file here
    
    % high-pass filter
    S.hpf = hpf_threshold;
    
    % put into batch
    fmri_spec.sess(i_run) = S;
    clear S
    
end

% remaining settings
fmri_spec.fact = struct('name', {}, 'levels', {});
fmri_spec.bases.hrf.derivs = [0 0];
fmri_spec.volt = 1;
fmri_spec.global = 'None';
fmri_spec.mthresh = 0.8; % -Inf
fmri_spec.mask = brain_mask;
fmri_spec.cvi = 'AR(1)';

% put into batch
matlabbatch{1}.spm.stats.fmri_spec = fmri_spec;

% save
save(fullfile(output_folder,'fmri_spec.mat'),'fmri_spec');

%% fMRI model estimation

% % review
% matlabbatch{2}.spm.stats.review.spmmat =  {fullfile(output_folder, 'SPM.mat')};
% matlabbatch{2}.spm.stats.review.display.orth = 1;
% matlabbatch{2}.spm.stats.review.print = 'png';

% estimate
matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(output_folder, 'SPM.mat')};
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

%% run batch
spm_jobman('run', matlabbatch);

end

function S = add_action_pmods(S,behav,idx_cond,idx_subselection)

% extract actions
var_names = {'a_subj','a_opp'};
chosen_actions(:,1) = behav.a_subj; % during choice
chosen_actions(:,2) = behav.a_opp; % during feedback

% check subselection indices
if nargin < 4
    idx_subselection = logical(ones(size(behav.a_subj,1),2));
elseif size(idx_subselection,2) == 1
    idx_subselection = repmat(idx_subselection,1,2);
end

% add pmods
for i_c = 1:2
    if isfield(S.cond(idx_cond(i_c)),'pmod')
        pmod_count = size(S.cond(idx_cond(i_c)).pmod,2); % identify # of existing pmods
    else
        pmod_count = 0;
    end
    idx_all = [];
    actions = unique(chosen_actions(:,1)); 
    actions(isnan(actions)) = []; % remove missings
    for i_a = 1:numel(actions)-1 % two dummies (3 is perfectly anticorrelated with 1+2; goes into intercept/cond)
        idx_played = double(chosen_actions(:,i_c) == actions(i_a)); % missings are automatically excluded
        idx_curr = idx_played(idx_subselection(:,i_c)) - mean(idx_played(idx_subselection(:,i_c)));
        if any(idx_curr) && ~any(all(spm_orth([idx_all idx_curr]) == 0)) % don't include non-orthogonal ones
            pmod_count = pmod_count + 1;
            S.cond(idx_cond(i_c)).pmod(pmod_count).name = sprintf('%s_%i',var_names{i_c},actions(i_a));
            S.cond(idx_cond(i_c)).pmod(pmod_count).param = idx_curr - mean(idx_curr);
            S.cond(idx_cond(i_c)).pmod(pmod_count).poly = 1;
            idx_all(:,end+1) = idx_curr;
        end
    end
    if pmod_count == 0
        S.cond(idx_cond(i_c)).pmod = struct('name', {}, 'param', {}, 'poly', {}); % to avoid SPM error
    end
end
                    
end
