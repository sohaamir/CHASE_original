function BAKR_2024_decode_levels(config,~)

RUN_PERMUTATION_TEST = 0;

decoding_types = {'wholebrain'};
timeperiods = {'ch','fb'}; % 'ch' or 'fb'
level_combos = {{'k1','k2','k3'}};

n_combos = numel(level_combos);

% unpack
folders = config.folders;
model_type = config.type;

% get first-level folder
folder_1L = fullfile(folders.results,model_type,'1L');

for i_t = 1:numel(timeperiods)
    
    timeperiod = timeperiods{i_t};
    
    for i_combo = 1:n_combos

        levels = level_combos{i_combo};

        % output folder
        output_dir_parent = fullfile(folders.results,model_type,'decoding_levels');
        if ~exist(output_dir_parent,'dir')
            mkdir(output_dir_parent);
        end

        % output sub-folder, if levels (to allow for different combinations) OR if decoding run numbers
        output_dir_parent = fullfile(output_dir_parent,sprintf('%s_%s',timeperiod,strjoin(levels,'_')));
        if ~exist(output_dir_parent,'dir')
            mkdir(output_dir_parent);
        end

        files = dir(folder_1L);
        subj_folders = {files.name};
        subj_folders(~startsWith(subj_folders,'sub-')) = [];

        %% collect files and create design matrix

        files_file = fullfile(output_dir_parent,'files.mat');

        % get correct label names
        switch model_type
            case 'pmods_x_level_x_outcome', labelnames = strcat(timeperiod,'_all_',levels);
            otherwise labelnames = strcat(timeperiod,'_',levels);
        end

        % get label numbers
        labels = linspace(-1,1,numel(labelnames));

        % put together new design struct
        fprintf('\nAssembling design... \n');

        % loop over subjects to collect files and labels
        i_subj_included = 0;
        for i_subj = 1:numel(subj_folders)

            fprintf('\nsubj %i... ',i_subj);
            beta_dir = fullfile(folder_1L,subj_folders{i_subj});
            % load(fullfile(beta_dir,'SPM.mat'),'SPM')

            % get and concatenate cfg
            try
                cfg = decoding_defaults;
                regressor_names = design_from_spm(beta_dir);
                cfg = decoding_describe_data(cfg,labelnames,labels,regressor_names,beta_dir);
                % cfg.design = make_design_cv(cfg);
            catch e % try reduced set (without missing labels)
                fprintf(2,'Skipping %s (not all levels found).',subj_folders{i_subj});
                continue
            end

            i_subj_included = i_subj_included + 1;
            cfg.files.chunk = ones(size(cfg.files.chunk))*i_subj_included; 

            if i_subj_included == 1 % copy all cfg from first
                cfg_all = cfg;
            else % for all others, append new information
                cfg_all.files.name = [cfg_all.files.name; cfg.files.name];
                cfg_all.files.chunk = [cfg_all.files.chunk; cfg.files.chunk];
                cfg_all.files.label = [cfg_all.files.label; cfg.files.label];
                cfg_all.files.labelname = [cfg_all.files.labelname; cfg.files.labelname];
                cfg_all.files.descr = [cfg_all.files.descr cfg.files.descr];
            end

        end
        cfg = cfg_all;

        save(files_file,'cfg');
        subj_file = fullfile(output_dir_parent,sprintf('%i_subj_%i_files.txt',i_subj_included,numel(cfg.files.name)));
        fclose(fopen(subj_file,'w'));

        cfg.design = make_design_cv(cfg);

        %% what to decode
        warning('off','decoding_write_results:overwrite_results');

        %% decoding settings

        cfg.results.overwrite = 1;

        cfg.verbose = 1; % 0,1,2 for increasing info
        cfg.plot_design = 0;
        cfg.software = spm('ver');
        cfg.plot_selected_voxels = 0; % <- for ROIs; put a number to plot searchlights every Xth step
        cfg.scale.method = 'min0max1global'; % <- for speedup, libsvm recommends this (get warning if not turned on)

        for i_type = 1:numel(decoding_types) % decode *across* subjects

            % i_type = 1;
            
            %% decoding
            decoding_type = decoding_types{i_type};
            fprintf('%s... ',decoding_type);

            cfg.analysis = decoding_type;

            % create output folder
            curr_output_child = fullfile(output_dir_parent,decoding_type);
            if ~exist(curr_output_child,'dir')
                mkdir(curr_output_child);
            end
            cfg.results.dir = curr_output_child;

            % add ROIs
            ROI_dir = fullfile(folders.project,'masks');
            if strcmp(decoding_type,'roi')
                ROIs = dir(ROI_dir);
                ROIs([ROIs.isdir]) = [];
                cfg.files.mask = fullfile(ROI_dir,{ROIs.name}');
            else
                cfg.files.mask = fullfile(ROI_dir,'gm.nii');
            end

            % type-specific settings
            cfg.results.write = 2; % 1=img and mat, 2=only mat

            % desired output
            outputs = {'accuracy_minus_chance','balanced_accuracy_minus_chance','decision_values','predicted_labels','true_labels',...
                           'confusion_matrix','SVM_weights'}; % ,'AUC_matrix','AUC_pairwise'
            [cfg.results.output,cfg.results.resultsname] = deal(outputs);
            cfg.design.unbalanced_data = 'ok';

            cfg.scale.check_datatrans_ok = 1;


            %% run decoding
            [results, cfg] = decoding(cfg);

            %% get permutation distribution
            if RUN_PERMUTATION_TEST
                
                n_perms = 5000;
                acc_perm = NaN(n_perms,1);
                conf_perm = NaN(numel(labels),numel(labels),n_perms);
                parfor i_perm = 1:n_perms
                    fprintf('\n%i/%i',i_perm,n_perms);
                    curr_perm = cfg;
                    curr_perm.results.write = 0;
                    for i_subj = 1:max(cfg.files.chunk)
                        idx = find(cfg.files.chunk == i_subj);
                        curr_perm.design.label(idx,:) = curr_perm.design.label(idx(randperm(numel(idx))),:); % permute within subjects
                    end
                    perm_result = decoding(curr_perm);
                    acc_perm(i_perm) = perm_result.balanced_accuracy_minus_chance.output;
                    conf_perm(:,:,i_perm) = perm_result.confusion_matrix.output{1};
                end
                curr_name = sprintf('permutation_distribution');
                save(fullfile(curr_output_child,curr_name),'acc_perm','conf_perm');
                
            end


        end

    end

end
        