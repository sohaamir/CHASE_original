function BAKR_2024_2L(config,subjects,con_names,i_con)
    
% unpack
folders = config.folders;
dataset = config.dataset;
model_type = config.type;
stat_models = config.stat_models;

n_subj = numel(subjects);

%% get files

% find files
file_name = sprintf('con-%i_%s',i_con,con_names{i_con});

% check if all subjects/contrast files available
first_level_dirs = dir(fullfile(folders.results,model_type,'1L','sub-*')); % sub-
idx_missing = (~contains(subjects,{first_level_dirs.name}));
if any(idx_missing)
    warning('First level folder missing for %i subjects (%s).',sum(idx_missing),strjoin(subjects(idx_missing),', '));
    subjects(idx_missing) = [];
end

% check if excess files (because subjects were excluded)
if numel(first_level_dirs) > numel(subjects)
    warning('Excluding %i subjects with existing first-level data.',numel(first_level_dirs) - numel(subjects));
    first_level_dirs(numel(subjects)+1:end) = [];
end

contrast_files = fullfile({first_level_dirs.folder}', subjects, sprintf('con_%04d.nii',i_con));
idx_missing = cellfun(@(x) ~exist(x,'file'),contrast_files);
if any(idx_missing)
    warning('Contrast files missing for %i subjects (%s).',sum(idx_missing),strjoin(subjects(idx_missing),', '));
    contrast_files(idx_missing) = [];
    subjects(idx_missing) = [];
end

%% get clinical data (and make sure it's in the correct order) <<<- should only be done if needed
if strcmp(dataset,'replication') %any(strcmp(stat_models,'regress')) 
    load(fullfile(folders.project,'data','replication','replication_subjects.mat'),'subj_data');
    data = subj_data;
    control_vars = {'age','sex'};
else
    control_vars = {};
end

%% non-parametric inference
    
np_types = config.voi;

% settings
n_permutations = 10000;
fwe_rate = 0.05;
cluster_forming_thrs = 0.01;

% ROIs
ROI_dir = fullfile(folders.project,'masks');
files = dir(ROI_dir);
files([files.isdir]) = [];
all_ROIs = {files.name};

model.ttest = {'OneSampT','MultiSub: One Sample T test on diffs/contrasts','snpm_bch_ui_OneSampT'};
model.regress = {'Corr','MultiSub: Simple Regression; 1 covariate of interest','snpm_bch_ui_Corr'};

% loop over statistical models (ttest, regression, etc)
for i_stat = 1:numel(stat_models)

    curr_model = model.(stat_models{i_stat});

    % loop over subsets of data (wholebrain, social, ROIs)
    for i_type = 1:numel(np_types)

        curr_type = np_types{i_type};

        % output folder
        output_folder = fullfile(folders.results,model_type,sprintf('2L_%s_np',stat_models{i_stat}),curr_type,file_name);

        % get correct subsets
        ROIs = all_ROIs;
        switch curr_type
            case 'wholebrain', ROIs = NaN;
            case 'socialbrain', ROIs(~strcmp(ROIs,'gm_social.nii')) = []; % remove all non-union ROIs
            case 'ROIs', ROIs(startsWith(ROIs,'gm')) = [];
            case 'previous_clusters'
                curr_con_parts = strsplit(con_names{i_con},'_');
                ROIs = {sprintf('%s_ALL_BINARIZED.nii',strjoin(curr_con_parts(1:end-2),'_'))};
        end

        for i_roi = 1:numel(ROIs)

            clear matlabbatch
            switch curr_type
                case 'ROIs', curr_output_folder = fullfile(output_folder,ROIs{i_roi}(1:end-4));
                otherwise, curr_output_folder = output_folder;
            end

            % ----------------------- factorial design ----------------------- %

            clear spec
            spec.DesignName = curr_model{2};
            spec.DesignFile = curr_model{3};

            switch stat_models{i_stat}

                case {'ttest','regress'}
                    spec.P = contrast_files;
                    spec.cov = struct('c', {}, 'cname', {});
                    spec.globalm.gmsca.gmsca_no = 1;
                    spec.globalm.glonorm = 1;

                    if strcmp(stat_models{i_stat},'regress')
                        spec.CovInt = data.(cov_of_interest)' - mean(data.(cov_of_interest)); % 1-by-x
                    end
                    
                    % all: control variables
                    if exist('control_vars','var') && ~isempty(control_vars) % <- also for ttest?
                        for i_var = 1:numel(control_vars)
                            spec.cov(i_var).cname = control_vars{i_var};
                            spec.cov(i_var).c = data.(control_vars{i_var}) - mean(data.(control_vars{i_var})); % x-by-1
                        end
                    else
                        spec.cov = struct('c', {}, 'cname', {});
                    end

            end

            spec.dir = {curr_output_folder};
            spec.nPerm = n_permutations;
            spec.vFWHM = [0 0 0];
            spec.bVolm = 1;
            spec.masking.tm.tm_none = 1;
            spec.masking.im = 1;
            spec.globalc.g_omit = 1;

            switch curr_type
                case 'ROIs', spec.ST.ST_none = 0; % <<< for voxel-level
                otherwise, spec.ST.ST_later = -1; % <<< for cluster-level
            end

            % define mask (if ROI used)
            switch curr_type
                case 'wholebrain', spec.masking.em = {''};
                case 'previous_clusters', spec.masking.em = {fullfile(ROI_dir,'replication',[ROIs{i_roi} ',1'])};
                otherwise, spec.masking.em = {fullfile(ROI_dir,[ROIs{i_roi} ',1'])};
            end

            i_m = 1;
            matlabbatch{i_m}.spm.tools.snpm.des.(curr_model{1}) = spec;

            % --------------------------- compute ---------------------------- %

            i_m = i_m + 1;
            dep_name = sprintf('%s: SnPMcfg.mat configuration file',curr_model{2});
            matlabbatch{i_m}.spm.tools.snpm.cp.snpmcfg(1) = cfg_dep(dep_name, substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','SnPMcfg'));

            % -------------------------- inference --------------------------- %

            for sign = [1 -1]

                % if cluster-level inference, loop over desired thresholds
                switch curr_type
                    case 'ROIs', n_thrs = 1;
                    otherwise, n_thrs = numel(cluster_forming_thrs);
                end

                for i_thr = 1:n_thrs 

                    i_m = i_m + 1;
                    matlabbatch{i_m}.spm.tools.snpm.inference.SnPMmat(1) = cfg_dep('Compute: SnPM.mat results file', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','SnPM'));

                    switch curr_type

                        % voxel-level inference for ROIs
                        case 'ROIs' 
                            switch sign
                                case 1, fname = sprintf('%s_%s_POS',file_name,ROIs{i_roi}(1:end-4));
                                case -1, fname = sprintf('%s_%s_NEG',file_name,ROIs{i_roi}(1:end-4));
                            end
                            matlabbatch{i_m}.spm.tools.snpm.inference.Thr.Vox.VoxSig.FWEth = fwe_rate;

                        % cluster-level inference otherwise (wholebrain or all of socialbrain)
                        otherwise 
                            cluster_forming_thr = cluster_forming_thrs(i_thr);
                            CFth = num2str(cluster_forming_thr);
                            switch sign
                                case 1, fname = sprintf('%s_%s_POS',con_names{i_con},CFth(3:end));
                                case -1, fname = sprintf('%s_%s_NEG',con_names{i_con},CFth(3:end));
                            end
                            matlabbatch{i_m}.spm.tools.snpm.inference.Thr.Clus.ClusSize.CFth = cluster_forming_thr;
                            matlabbatch{i_m}.spm.tools.snpm.inference.Thr.Clus.ClusSize.ClusSig.FWEthC = fwe_rate;

                    end

                    matlabbatch{i_m}.spm.tools.snpm.inference.Tsign = sign;
                    matlabbatch{i_m}.spm.tools.snpm.inference.WriteFiltImg.name = fname;
                    matlabbatch{i_m}.spm.tools.snpm.inference.Report = 'MIPtable';

                    i_m = i_m + 1;
                    matlabbatch{i_m}.spm.util.print.fname = fname;
                    matlabbatch{i_m}.spm.util.print.fig.figname = 'Graphics';
                    matlabbatch{i_m}.spm.util.print.opts = 'fig';

                end

            end

            % run
            spm_jobman('run',matlabbatch);

        end
    end
end

%% save sample stats

n_files = numel(contrast_files);
sample_info = sprintf('%i_of_%i_subjects.txt',n_files,n_subj);
fclose(fopen(fullfile(output_folder,sample_info),'w'));
if n_subj == n_files
    fprintf('All subjects included (%i/%i).\n\n',n_files,n_subj);
else
    fprintf(2,'%i subjects missing (only %i/%i found).\n\n',n_subj-n_files,n_files,n_subj);
end

end
