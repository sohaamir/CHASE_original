function BAKR_2024_create_nuisance_regressors(nuisance_file,folders,subj,i_run)
 
behav_folder = fullfile(folders.prepro,subj,'ses-00001','beh');
runs_folder = fullfile(folders.prepro,subj,'*','func');

% get motion correction regressors
file_motion = dir(fullfile(runs_folder,sprintf('%s*_run-%i*confounds_timeseries.tsv',subj,i_run)));
motion_all = tdfread(fullfile(file_motion.folder,file_motion.name));

vars = {'trans_x','trans_y','trans_z','rot_x','rot_y','rot_z'};
for i_var = 1:numel(vars)
    curr_var = vars{i_var};
    motion_all.([curr_var '_deriv'])(1,1) = 0;
    for ii = 2:size(motion_all.trans_x,1)
        motion_all.([curr_var '_deriv'])(ii,1) = str2double(motion_all.([curr_var '_derivative1'])(ii,:));
    end
end

% extract the 6 std motion params & their derivates, and the global signal
motion_vars = {'trans_x','trans_y','trans_z','rot_x','rot_y','rot_z',...
               'trans_x_deriv','trans_y_deriv','trans_z_deriv','rot_x_deriv','rot_y_deriv','rot_z_deriv',...
               'global_signal'};
motion_param = [];
for i_var = 1:numel(motion_vars)
    motion_param = [motion_param motion_all.(motion_vars{i_var})];
end

% physiological correction
physioR_file = fullfile(behav_folder,'physio',sprintf('RegPhysio_%s_run_%i.mat',subj,i_run));
if exist(physioR_file, 'file')
    out = load(physioR_file);
    physio = out.physio;
else
    physio.model.R_column_names = [];
    physio.model.R = [];    
end
physio_names = cellstr(strcat('physio_',num2str([1:size(physio.model.R,2)]')));
physio_names = cellfun(@(x) strrep(x,' ',''),physio_names,'UniformOutput',0);
physio_param = physio.model.R;

% combine both to nuisance regressors matrix
motion_names = strcat('motion_',motion_vars);
names = [motion_names'; physio_names];
if size(physio_param,1) > size(motion_param,1)
    physio_param(size(motion_param,1)+1:end,:) = []; % if more physio recordings than scans, discard those
    warning('Removing excess physio recordings (subj %s, run %i).',subj,i_run);
end
R = [motion_param physio_param];

% save file
save(nuisance_file,'names','R');

end
