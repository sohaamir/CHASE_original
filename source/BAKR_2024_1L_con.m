function con_names = BAKR_2024_1L_con(config,subj,flag_only_names)

delete_old_con = true;
con_to_delete = [];

% spm('defaults', 'fMRI');
% spm_jobman('initcfg');

% unpack
folders = config.folders;
model_type = config.type;
pmods = config.pmods;

% get files
output_folder = fullfile(folders.results,model_type,'1L',subj);
load(fullfile(output_folder,'SPM.mat'),'SPM');

        
%% sub-selecting runs
valid_runs = {'Sn('}; % i.e. all of them

%% get phases/outcomes/pmods (or their combination)
regs = SPM.xX.name';
exp_phases = {'ch','fb'};
        
switch model_type
        
    case 'pmods_whole_run'
        
        % first the timing regressors (without pmod)
        for i_phase = 1:numel(exp_phases)
            curr_phase = exp_phases{i_phase};
            all_runs.(curr_phase) = double(contains(regs,sprintf(') %s*bf(1)',curr_phase))); %asds version
        end
        
        % then the pmods
        if isa(pmods,'char'), pmods = {pmods}; end
        for i_pmod = 1:numel(pmods)
            curr_pmod = pmods{i_pmod};
            target = sprintf('x%s^',curr_pmod);
            all_runs.(curr_pmod) = double(contains(regs,target) & contains(regs,valid_runs) & endsWith(regs,'(1)'));
        end
        
end

%% identify runs for subsets
specific_runs.all = ones(size(regs));

%% loop over subsets
targets = fieldnames(all_runs);
subsets = fieldnames(specific_runs);
for i_target = 1:numel(targets)
    for i_sub = 1:numel(subsets)
        cons.(strcat(targets{i_target},'_',subsets{i_sub})) = all_runs.(targets{i_target}) .* specific_runs.(subsets{i_sub});
    end
end

%% create main contrasts
% rescale to sum to one (and identify empty ones to delete later)
i_con = 1;
for field = fieldnames(cons)'
    if sum(cons.(field{1})) > 0
        cons.(field{1}) = cons.(field{1}) / sum(cons.(field{1}));
    else
        cons.(field{1}) = ones(size(regs));  % <- will be deleted at the end of the script
        fprintf(2,'\nWill delete contrast #%i (%s) for %s (no valid regressors).',i_con,field{1},subj);
        con_to_delete(end+1) = i_con;
    end
    i_con = i_con + 1;
end

% put into matlabbatch
matlabbatch{1}.spm.stats.con.spmmat = {fullfile(output_folder, 'SPM.mat')};
con_names = fieldnames(cons);
for i_con = 1:numel(con_names)
    curr_con = con_names{i_con};
    matlabbatch{1}.spm.stats.con.consess{i_con}.tcon.name = curr_con;
    matlabbatch{1}.spm.stats.con.consess{i_con}.tcon.weights = cons.(curr_con);
    matlabbatch{1}.spm.stats.con.consess{i_con}.tcon.sessrep = 'none';
end
matlabbatch{1}.spm.stats.con.delete = delete_old_con;

%% create additional contrasts
% add all inverse contrasts
for i_con = 1:numel(con_names)
    original_weights = matlabbatch{1}.spm.stats.con.consess{i_con}.tcon.weights;
    matlabbatch{1}.spm.stats.con.consess{numel(con_names)+i_con}.tcon.name = strcat(con_names{i_con},'_INV');
    matlabbatch{1}.spm.stats.con.consess{numel(con_names)+i_con}.tcon.weights = -original_weights;
    matlabbatch{1}.spm.stats.con.consess{numel(con_names)+i_con}.tcon.sessrep = 'none';
end
con_names = [con_names; strcat(con_names,'_INV')];

contrast_vector = contains(regs,'*bf(1)') & contains(regs,valid_runs) & ~contains(regs,{'missing-','a_subj_','a_opp_'});
matlabbatch{1}.spm.stats.con.consess{numel(con_names)+1}.fcon.name = 'Effects of interest';
matlabbatch{1}.spm.stats.con.consess{numel(con_names)+1}.fcon.weights = double(contrast_vector)';
matlabbatch{1}.spm.stats.con.consess{numel(con_names)+1}.fcon.sessrep = 'none';
        
%% run
if ~flag_only_names
    
    % run
    spm_jobman('run',matlabbatch); 
    
    % check if contrast names are correct in SPM file (i.e. correspond to function output)
    load(fullfile(output_folder,'SPM.mat'),'SPM');
    assert(all(strcmp({SPM.xCon(1:end-1).name},con_names')),'Contrast names don''t match.');

    % remove invalid contrast files (and put placeholders into SPM file)
    if numel(con_to_delete) > 0
        fprintf(2,'Deleting contrasts %s for %s (no valid regressors).\n\n',strjoin(cellstr(num2str(con_to_delete')),', '),subj);
        for ii = 1:numel(con_to_delete)
            idx = con_to_delete(ii);
            idx(2) = find(strcmp(con_names,[con_names{idx} '_INV']));
            for jj = idx
                delete(fullfile(output_folder,sprintf('con_%04d.nii',jj)));
                delete(fullfile(output_folder,sprintf('spmT_%04d.nii',jj)));
                SPM.xCon(jj).name = 'N/A';
                SPM.xCon(jj).c = NaN(size(regs));
            end
        end
        save(fullfile(output_folder,'SPM.mat'),'SPM')
    end  

    % check if all placeholder contrasts have successfully been deleted
    files = dir(output_folder);
    con_files = files(contains({files.name},'con_0'));
    assert(numel(con_names)-numel(con_to_delete)*2 == numel(con_files));
    
end

end
