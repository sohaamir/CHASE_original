function BAKR_2024_decode_BU(config,~,type)

BOOTSTRAP_WEIGHTS = 1;
RUN_PERMUTATION_TEST = 1;

% unpack
folders = config.folders;
model_type = config.type;
pmod = config.pmods;
dataset = config.dataset;

% settings
timeperiods = {'fb'};
DO_ROI = 1;

% get files
folder_1L = fullfile(folders.results,model_type,'1L');
files = dir(folder_1L);
subj_folders = {files.name};
subj_folders(~startsWith(subj_folders,'sub-')) = [];
    
%% get predictor vars
if strcmp(dataset,'replication')
    load(fullfile(folders.project,'data','replication','replication_subjects.mat'),'subj_data');
end
    
%% gather relevant files
for i_t = 1:numel(timeperiods)
    
    timeperiod = timeperiods{i_t};

    % output folder
    output_dir_parent = fullfile(folders.results,model_type,'decoding_belief_updates');
    if ~exist(output_dir_parent,'dir')
        mkdir(output_dir_parent);
    end

    % output sub-folder, if levels
    output_dir_parent = fullfile(output_dir_parent,timeperiod);
    output_dir_parent = [output_dir_parent '_meanbetas'];
    if ~exist(output_dir_parent,'dir')
        mkdir(output_dir_parent);
    end

    %% collect files and create design matrix

    files_file = fullfile(output_dir_parent,'files.mat');

    % get correct label names
    bins = {'1','2','3','4','5'};
    labelnames = strcat(timeperiod,'_',pmod,'_',bins);
    labels = linspace(-1,1,numel(labelnames));

    % put together new design struct
    fprintf('\nAssembling design... \n');

    cfg.files.name = {};
    cfg.files.label = [];
    cfg.files.chunk = [];
    cfg.files.runs = [];
    idx_included = zeros(numel(subj_folders),1);

    % loop over subjects to collect files and labels
    i_subj_included = 0;
    for i_subj = 1:numel(subj_folders)

        % try to get SPM file
        fprintf('\nsubj %i... ',i_subj);
        SPM_file = fullfile(folder_1L,subj_folders{i_subj},'SPM.mat');
        if exist(SPM_file,'file')
            load(SPM_file,'SPM');
        else
            fprintf(2,'No SPM file - skipping.');
            continue
        end

        % find valid regressors
        regressor_names = SPM.xX.name';
        idx_beta = find(contains(regressor_names,strcat(labelnames,'*bf(1)')));

        if any(idx_beta)
            i_subj_included = i_subj_included + 1;
            new_runs = cellfun(@(x) str2double(x(4)),regressor_names(idx_beta));
            new_labels = cellfun(@(x) str2double(x(end-6)),regressor_names(idx_beta));
            new_names = fullfile(folder_1L,subj_folders{i_subj},cellstr(strcat('beta_',num2str(idx_beta,'%04d'),'.nii')));
            [~,idx] = sort(new_labels);
            cfg.files.name = [cfg.files.name; new_names(idx)];
            cfg.files.label = [cfg.files.label; new_labels(idx)];
            cfg.files.runs  = [cfg.files.runs; new_runs(idx)];
            cfg.files.chunk = [cfg.files.chunk; ones(numel(new_names),1)*i_subj_included];
            idx_included(i_subj) = 1;
        else
            fprintf(2,'No matching regressors found - skipping.');
        end

    end

    % rescale to [-1,1]
    cfg.files.label = (cfg.files.label-1)/2 - 1;
    cfg.idx_included = idx_included;

    save(files_file,'cfg');
    subj_file = fullfile(output_dir_parent,sprintf('%i_subj_%i_files.txt',i_subj_included,numel(cfg.files.name)));
    fclose(fopen(subj_file,'w'));

    %% decoding settings
    curr_output_child = output_dir_parent;

    %% get average betas
    ii = 1;
    if ~exist(fullfile(output_dir_parent,'mean_images'),'dir')
        mkdir(fullfile(output_dir_parent,'mean_images'));
    end
    for i_chunk = 1:max(cfg.files.chunk) % 42,59
        idx_subj = (cfg.files.chunk == i_chunk);
        fprintf('\n\nSubj %d... \n',i_chunk);
        for label = labels
            if any(label == unique(cfg.files.label(idx_subj))) % skip missings
                idx = (idx_subj & cfg.files.label == label);
                file_name = sprintf('subj_%i_bin_%i_mean.nii',i_chunk,find(label == labels));

                if ~isfile(fullfile(output_dir_parent,'mean_images',file_name))
                    one_subj_one_label = fmri_data(cfg.files.name(idx),'noverbose');
                    one_subj_one_label = one_subj_one_label.mean;
                    one_subj_one_label.fullpath = fullfile(output_dir_parent,'mean_images',file_name);
                    one_subj_one_label.write('overwrite');
                end

                mean_files{ii} = fullfile(output_dir_parent,'mean_images',file_name);
                mean_chunks(ii) = i_chunk;
                mean_labels(ii) = label;
                ii = ii + 1;
            end
        end
    end
    cfg.files.name  = mean_files';
    cfg.files.chunk = mean_chunks';
    cfg.files.label = mean_labels';

    %% train or predict
    switch type

        case 'predict'

            % main pattern (without regularization)
            pred_file = fullfile(curr_output_child,'predictions','pred_meanbetas.mat');
            [r,BU_hat,rmse] = BAKR_2024_apply_pattern(fullfile(output_dir_parent,'mean_images'),folders.project);
            if ~isfolder(fullfile(curr_output_child,'predictions'))
                mkdir(fullfile(curr_output_child,'predictions'));
            end
            save(pred_file,'r','BU_hat','rmse');
            tanh(mean(atanh(r)))
            
            data        = subj_data;
            data.BU_hat = BU_hat;
            data.rmse   = rmse;
            data.r      = r;
            save(fullfile(folders.project,'results','decoding','decoding_replication.mat'),'data');

            pattern_folder = fullfile(folders.project,'results','pmod_ordinal','decoding_belief_updates','fb_meanbetas','train_all','control_patterns');
            models = {'no regularization','nested within subj','nested 5-fold','all 5-fold'};
            for i_model = 1:4
                pattern_file = fullfile(pattern_folder,sprintf('model_%i.nii',i_model));
                r(:,i_model) = BAKR_2024_apply_pattern(fullfile(output_dir_parent,'mean_images'),folders.project,pattern_file);
            end

            pred_file_controls = fullfile(curr_output_child,'predictions','pred_control_analyses.mat');
            save(pred_file_controls,'r');

        case 'predict_reverse'

            % main pattern (without regularization)
            pattern_file = fullfile(folders.project,'results','replication','decoding','pattern_replication.nii');
            [r,BU_hat,rmse] = BAKR_2024_apply_pattern(fullfile(output_dir_parent,'mean_images'),folders.project,pattern_file);
            if ~isfolder(fullfile(curr_output_child,'predictions'))
                mkdir(fullfile(curr_output_child,'predictions'));
            end
            pred_file = fullfile(curr_output_child,'predictions','pred_meanbetas.mat');
            save(pred_file,'r','BU_hat','rmse');
            tanh(mean(atanh(r)))
              
        case 'train'

            masks = {'brainmask_canlab.nii'};

            if DO_ROI
                ROI_dir = fullfile(folders.project,'masks');
                ROIs = dir(ROI_dir);
                ROIs([ROIs.isdir]) = [];
                new_masks = fullfile(ROI_dir,{ROIs.name}');
                masks = [masks; new_masks];
                labels = [{'wholebrain'} cellfun(@(x) x(1:end-4),{ROIs.name},'UniformOutput',0)];
            else
                labels = {'wholebrain'};
            end

            % remove canlab mask
            masks(1) = [];
            labels(1) = [];

            sub_name = 'train_all';
            curr_output_child = fullfile(output_dir_parent,sub_name);
            if ~exist(curr_output_child,'dir')
                mkdir(curr_output_child);
            end

            %% === main (unregularized) models ===
            for i_mask = 1:numel(masks)
                data = fmri_data(cfg.files.name,masks{i_mask});

                data.Y = cfg.files.label;
                [cverr, stats, opt] = predict(data,'algorithm_name','cv_lassopcr','nfolds',cfg.files.chunk);
                curr_filename = sprintf('canlab_%s.mat',labels{i_mask});
                save(fullfile(curr_output_child,curr_filename),'cverr','stats','opt');

                for i_subj = 1:numel(stats.teIdx)
                    idx = stats.teIdx{i_subj};
                    [rs(i_subj),ps(i_subj)] = corr(stats.yfit(idx),stats.Y(idx));
                end
                rs_all(i_mask,:) = rs;
                r_average(i_mask) = tanh(mean(atanh(rs)));
                r_sem(i_mask) = std(rs)/sqrt(numel(rs));
                r_overall(i_mask) = stats.pred_outcome_r;

                if strcmp(labels{i_mask},'gm') % save for fingerprint (below)
                    stats_gm = stats;
                end

            end

            if DO_ROI, curr_filename = 'ROIs';
            else, curr_filename = 'wholebrain';
            end

            save(fullfile(curr_output_child,sprintf('canlab_%s_r.mat',curr_filename)),'r_overall','rs_all','r_average','r_sem');

            %% === alternative (hyperparameter-tuned) models ===

            i_mask = find(endsWith(masks,'gm.nii'));
            models = {'no regularization','nested within subj','nested 5-fold','all 5-fold'};

            data = fmri_data(cfg.files.name,masks{i_mask});
            data.Y = cfg.files.label;

            % alternative models 
            n_folds = 5;
            [~,edges] = histcounts(cfg.files.chunk,n_folds);
            nested_chunks = sum(cfg.files.chunk < edges,2);
            teIdx = stats.teIdx;

            figure; hold on;
            for i_model = 1:4

                % fit
                switch i_model
                    case 1, stats = stats_gm; teIdx = stats.teIdx;
                    case 2, [~,stats] = predict(data,'algorithm_name','cv_lassopcr','nfolds',cfg.files.chunk,'EstimateParams'); % with nested CV to determine lambda
                    case 3, [~,stats] = predict(data,'algorithm_name','cv_lassopcr','nfolds',cfg.files.chunk,'EstimateParams','NestedFolds',n_folds); % with nested CV to determine lambda
                    case 4, [~,stats] = predict(data,'algorithm_name','cv_lassopcr','nfolds',nested_chunks,'EstimateParams'); % same, but n-fold (instead of LOO-CV)
                end
                all_stats{i_model} = stats;
                all_stats{i_model}.type = models{i_model};

                % save pattern
                stats = all_stats{i_model};
                mkdir(fullfile(curr_output_child,'control_patterns'));
                output_file = fullfile(curr_output_child,'control_patterns',sprintf('model_%i.nii',i_model));
                write(stats.weight_obj,'fname',output_file,'overwrite');

            end

            save(fullfile(curr_output_child,'effect_of_regularization.mat'),'all_stats');

            %% bootstrapping weights
            if BOOTSTRAP_WEIGHTS
                
                n = 10000;
                idx_gm = find(strcmp(labels,'gm') | strcmp(labels,'whole brain'));
                data = fmri_data(cfg.files.name,masks{idx_gm});
                data.Y = cfg.files.label;
                [cverr, stats, opt] = predict(data,'algorithm_name','cv_lassopcr','nfolds',cfg.files.chunk,...
                                                   'bootweights',1,'bootsamples',n,'savebootweights',0,...
                                                   'useparallel',1);
                curr_filename = sprintf('bootstraps_%s_%i.mat',labels{idx_gm},n);
                save(fullfile(curr_output_child,curr_filename),'cverr','stats','opt','-v7.3');

                % apply FDR correction
                w = threshold(stats.weight_obj, .05, 'fdr'); 
                filename = sprintf('pattern_%s_FDR',pmod(1:end-3));
                fingerprint_thresh_dir = fullfile(folders.results,'decoding',sprintf('%s.nii',filename));
                fingerprint_binarized_dir = fullfile(folders.results,'decoding',sprintf('%s_binarized.nii',filename));
                mkdir(fullfile(folders.results,'decoding'));
                write(w,'thresh','fname',fingerprint_thresh_dir,'overwrite');
                binarize_nii(fingerprint_thresh_dir,fingerprint_binarized_dir);

                % save raw pattern
                filename = sprintf('pattern_%s',pmod(1:end-3));
                fingerprint_dir = fullfile(folders.results,'decoding',sprintf('%s.nii',filename));
                write(stats.weight_obj,'thresh','fname',fingerprint_dir,'overwrite');
                
            end

            %% permutation testing
            if RUN_PERMUTATION_TEST

                ROIs = dir(ROI_dir);
                ROIs([ROIs.isdir]) = [];
                masks = fullfile(ROI_dir,{ROIs.name}');
                labels = cellfun(@(x) x(1:end-4),{ROIs.name},'UniformOutput',0);

                idx_gm = find(strcmp(labels,'gm') | strcmp(labels,'whole brain')); % this is gm, as renamed above for the figure
                data = fmri_data(cfg.files.name,masks{idx_gm});
                data.Y = cfg.files.label;

                rng 'shuffle';
                permutation_folder = fullfile(curr_output_child,'permutations');
                if ~exist(permutation_folder,'dir')
                    mkdir(permutation_folder);
                end

                n_perms = 5000;
                [r,mse,rmse,meanabserr] = deal(NaN(n_perms,1));
                parfor i_perm = 1:n_perms

                    curr_data = data;
                    fprintf('\n%i/%i',i_perm,n_perms);
                    for i_subj = 1:max(cfg.files.chunk)
                        idx = find(cfg.files.chunk == i_subj);
                        curr_data.Y(idx,:) = curr_data.Y(idx(randperm(numel(idx))),:); % permute within subjects
                    end
                    [cverr, stats, opt] = predict(curr_data,'algorithm_name','cv_lassopcr','nfolds',cfg.files.chunk,'verbose',0)
                    r(i_perm) = stats.pred_outcome_r;
                    mse(i_perm) = stats.mse;
                    rmse(i_perm) = stats.rmse;
                    meanabserr(i_perm) = stats.meanabserr;

                    [rs,ps] = deal([]);
                    for i_subj = 1:numel(stats.teIdx)
                        idx = stats.teIdx{i_subj};
                        [rs(i_subj),ps(i_subj)] = corr(stats.yfit(idx),stats.Y(idx));
                    end
                    rs_all(i_perm,:) = rs;
                    r_average(i_perm) = tanh(mean(atanh(rs)));
                    r_sem(i_perm) = std(rs)/sqrt(numel(rs));

                end
                curr_name = 'permutation_distribution';
                save(fullfile(permutation_folder,curr_name),'r','mse','rmse','meanabserr','rs_all','r_average','r_sem');   

            end
            
            %% copy outputs (for easier access)
            mkdir(fullfile(folders.results,'decoding'));
            copyfile(fullfile(curr_output_child,'canlab_gm.mat'),...
                     fullfile(folders.results,'decoding','decoding.mat'));
            files = dir(fullfile(curr_output_child,'canlab_*'));
            files(contains({files.name},{'NAcc','Amy','nonsocial','bootstrap'})) = [];
            mkdir(fullfile(folders.results,'decoding','ROIs'));
            for i_file = 1:numel(files)
                copyfile(fullfile(curr_output_child,files(i_file).name),...
                         fullfile(folders.results,'decoding','ROIs',files(i_file).name));
            end

            % extract and save fingerprint
            filename = sprintf('fingerprint_%s.nii',pmod(1:end-3));
            fingerprint_dir = fullfile(folders.project,'pattern',filename);
            write(stats_gm.weight_obj,'fname',fingerprint_dir,'overwrite');

        case 'train_control'
              
            % get masks
            ROI_dir = fullfile(folders.project,'masks');
            ROIs = dir(ROI_dir);
            ROIs([ROIs.isdir]) = [];
            masks = fullfile(ROI_dir,{ROIs.name}');
            labels = cellfun(@(x) x(1:end-4),{ROIs.name},'UniformOutput',0);
            idx_gm = contains(masks,'gm');

            sub_name = 'train_control_analyses';
            curr_output_child = fullfile(output_dir_parent,sub_name);
            if ~exist(curr_output_child,'dir')
                mkdir(curr_output_child);
            end

            % decode (based on all masks)
            labels{end+1} = 'ALL_COMBINED';
            for i_mask = 1:numel(masks)+1

                if i_mask <= numel(masks)
                    data = fmri_data(cfg.files.name,masks{i_mask});
                    [data.dat,dat(i_mask,:)] = deal(mean(data.dat,1));
                    algo = 'cv_univregress';
                else
                    data.dat = dat(~idx_gm,:);
                    algo = 'cv_multregress';
                end

                data.Y = cfg.files.label;
                [cverr, stats, opt] = predict(data,'algorithm_name',algo,'nfolds',cfg.files.chunk);
                curr_filename = sprintf('single_ROI_%s.mat',labels{i_mask});
                save(fullfile(curr_output_child,curr_filename),'cverr','stats','opt');

                for i_subj = 1:numel(stats.teIdx)
                    idx = stats.teIdx{i_subj};
                    [rs(i_subj),ps(i_subj)] = corr(stats.yfit(idx),stats.Y(idx));
                end
                rs_all(i_mask,:) = rs;
                r_average(i_mask) = tanh(mean(atanh(rs)));
                r_sem(i_mask) = std(rs)/sqrt(numel(rs));
                r_overall(i_mask) = stats.pred_outcome_r;

            end

            save(fullfile(curr_output_child,'univariate_models.mat'),'rs_all','r_average','r_overall','labels');

        case 'train_reverse'

            ROI_dir = fullfile(folders.project,'masks');
            ROIs = dir(ROI_dir);
            ROIs = ROIs(strcmp({ROIs.name},'gm.nii'));
            masks = fullfile(ROI_dir,{ROIs.name}');
            labels = cellfun(@(x) x(1:end-4),{ROIs.name},'UniformOutput',0);

            sub_name = 'train_all';
            curr_output_child = fullfile(output_dir_parent,sub_name);
            if ~exist(curr_output_child,'dir')
                mkdir(curr_output_child);
            end

            % fit model
            for i_mask = 1:numel(masks)
                data = fmri_data(cfg.files.name,masks{i_mask});

                data.Y = cfg.files.label;
                [cverr, stats, opt] = predict(data,'algorithm_name','cv_lassopcr','nfolds',cfg.files.chunk);
                curr_filename = sprintf('canlab_%s.mat',labels{i_mask});
                save(fullfile(curr_output_child,curr_filename),'cverr','stats','opt');

            end

            %% bootstrapping weights
            if BOOTSTRAP_WEIGHTS
                
                n = 10000;
                idx_gm = find(strcmp(labels,'gm') | strcmp(labels,'whole brain'));
                data = fmri_data(cfg.files.name,masks{idx_gm});
                data.Y = cfg.files.label;
                [cverr, stats, opt] = predict(data,'algorithm_name','cv_lassopcr','nfolds',cfg.files.chunk,...
                                                   'bootweights',1,'bootsamples',n,'savebootweights',0,...
                                                   'useparallel',1);
                curr_filename = sprintf('bootstraps_%s_%i.mat',labels{idx_gm},n);
                save(fullfile(curr_output_child,curr_filename),'cverr','stats','opt','-v7.3');

                % apply FDR correction
                w = threshold(stats.weight_obj, .05, 'fdr'); 
                fingerprint_thresh_dir = fullfile(folders.results,'decoding','pattern_replication_FDR_0_05.nii');
                mkdir(fullfile(folders.results,'decoding'));
                write(w,'thresh','fname',fingerprint_thresh_dir,'overwrite');
                
            end

            % save raw pattern
            fingerprint_dir = fullfile(folders.results,'decoding','pattern_replication.nii');
            write(stats.weight_obj,'thresh','fname',fingerprint_dir,'overwrite');
            
            % copy outputs
            mkdir(fullfile(folders.results,'decoding'));
            copyfile(fullfile(curr_output_child,'canlab_gm.mat'),...
                     fullfile(folders.results,'decoding','decoding.mat'));

    end

end