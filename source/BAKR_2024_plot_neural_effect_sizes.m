function BAKR_2024_plot_neural_effect_sizes(project_folder,dataset,orientation)

if nargin < 2
    dataset = 'primary';
end
if nargin < 3
    orientation = 'horizontal';
end

folders.results = fullfile(project_folder,'results');
if strcmp(dataset,'replication')
    folders.results = fullfile(folders.results,'replication');
end

ROI_dir = fullfile(project_folder,'masks');

pmod = {'SV_ch','APE_fb','subj_KL_div_fb'};
model_type = 'pmods_whole_run';

con_names = {'ch_all','fb_all','SV_ch_all','subj_KL_div_fb_all','APE_fb_all'};

%% prep
colors = get_matlab_colors();

% identify folders
first_level_dirs = dir(fullfile(folders.results,model_type,'1L','sub-*'));
second_level_np_dirs = dir(fullfile(project_folder,'results',model_type,'2L_ttest_np','socialbrain','con-*'));

% get subject numbers
subjects = {first_level_dirs.name}';

% get ROIs
files = dir(ROI_dir);
files([files.isdir]) = [];
ROIs = {files.name};
ROI_names = cellfun(@(x) x(1:end-4),ROIs,'UniformOutput',0);

% move union to the end
idx_all = endsWith(ROI_names,'gm_social');
ROI_names = [ROI_names(~idx_all) ROI_names(idx_all)];
ROIs = [ROIs(~idx_all) ROIs(idx_all)];
idx_del = (strcmp(ROI_names,'gm') | strcmp(ROI_names,'gm_nonsocial'));
ROI_names(idx_del) = [];
ROIs(idx_del) = [];

for i_name = 1:numel(ROI_names)
    if contains(ROI_names{i_name},'_')
    	ROI_names{i_name} = sprintf('%s (%s)',ROI_names{i_name}(1:end-2),lower(ROI_names{i_name}(end)));           
    end
end
    
% create location matrices for ROIs
for i_roi = 1:numel(ROIs)
    curr_ROI = ROIs{i_roi}(1:end-4);
    ROI = fullfile(ROI_dir,ROIs{i_roi});
    Y = spm_read_vols(spm_vol(ROI),1);
    indx = find(Y>0);
    [x,y,z] = ind2sub(size(Y),indx);
    XYZ.(curr_ROI) = [x y z]';
end

%% loop over pmods
for i_pmod = 1:numel(pmod)
    
    curr_pmod = pmod{i_pmod};
    fprintf('\nExtracting %s... ',curr_pmod);
        
    % identify relevant contrasts
    i_con = find(contains(con_names,curr_pmod) & endsWith(con_names,'_all'));
    subj_x_roi = NaN(numel(subjects),numel(ROIs),1);

    % identify contrast file from each subject, and remove missing ones
    contrast_files = fullfile({first_level_dirs.folder}',subjects,sprintf('con_%04i.nii',i_con));
    idx_missing = cellfun(@(x) ~exist(x,'file'),contrast_files);
    contrast_files(idx_missing) = [];

    % loop over ROIs & extract mean data for each subject
    for i_roi = 1:numel(ROIs)

        curr_ROI = ROIs{i_roi}(1:end-4);

        % get current ROI
        curr_XYZ = XYZ.(curr_ROI);
        ROI_ind = sub2ind(size(Y),curr_XYZ(1,:),curr_XYZ(2,:),curr_XYZ(3,:));
        ROI_nii = zeros(65,77,56);
        ROI_nii(ROI_ind) = 1;

        % find 2L results files for current pmod and ROI
        dir_names = {second_level_np_dirs.name};
        idx = find(contains(dir_names,curr_pmod) & endsWith(dir_names,'_all'));
        assert(~isempty(idx),'Cannot find second level np folder for %s. (Maybe didn''t run it yet?)');
        curr_folder = fullfile(second_level_np_dirs(idx).folder,second_level_np_dirs(idx).name);
        curr_files = dir(fullfile(curr_folder,'*.nii'));
        curr_files = curr_files(contains({curr_files.name},'_01_')); % exclude binarized version

        assert(~isempty(curr_files),'Cannot find 2nd level ROI union file for given cluster-forming threshold.');

        % check for significant voxels within current ROI (positive or negative)
        Y = NaN(65,77,56,numel(curr_files));
        cluster_size = NaN(numel(curr_files));
        for i_outcome = 1:numel(curr_files)
            curr_file = fullfile(curr_files(i_outcome).folder,curr_files(i_outcome).name);
            Y_new = spm_read_vols(spm_vol(curr_file),1) > 0 & ROI_nii; % significant AND part of the ROI
            cluster_size(i_outcome) = nansum(Y_new(:) > 0);
            Y(:,:,:,i_outcome) = Y_new;
        end

        % check if any voxel within current ROI survive
        if ~any(cluster_size(:) > 0)
            continue
        end

        % create location matrix
        [~,idx_max] = max(cluster_size);
        Y = Y(:,:,:,idx_max);
        indx = find(Y>0);
        [x,y,z] = ind2sub(size(Y),indx);
        curr_XYZ = [x y z]';

        % extract average effect size within significant cluster for each subject
        for i_subj = 1:numel(contrast_files)
            curr_con = contrast_files{i_subj};
            subj_x_roi(i_subj,i_roi) = nanmean(spm_get_data(curr_con,curr_XYZ),2);
        end     

    end

    % assemble struct
    assert(~any(all(squeeze(all(isnan(subj_x_roi),2)),2)));
    avg_effect.(curr_pmod) = subj_x_roi;
    
end

% % save
% save('effects_within_sig_clusters.mat','pmod','avg_effect','ROIs','ROI_names');

%% plot (requires: pmod avg_effect ROIs ROI_names)

figure;
fprintf('\n\nPlotting... \n');
for i_pmod = 1:numel(pmod)
    
    curr_pmod = pmod{i_pmod};

    subj_x_roi = avg_effect.(curr_pmod);
    
    % test against zero
    mu(:,1) = nanmean(subj_x_roi(:,:));
    [H_new,~,CI_new] = ttest(subj_x_roi(:,:));
    H(:,1) = H_new;
    n = size(subj_x_roi,1);
    err = std(subj_x_roi(:,:))/sqrt(n);
    sem(:,:) = [mu(:,1)' + err;mu(:,1)' - err];

%     f = figure('Position',[600 400 1300 600]); hold on;
    switch orientation
        case 'horizontal', subplot(3,1,i_pmod); idx_del = [3,4,7,8];
        case 'vertical', subplot(1,3,i_pmod);  idx_del = [3,4,7,8,find(isnan(mu))'];
    end
    hold on;

    % select ROIs to plot
    % idx_del = [3,4,7,8];
    % idx_del = [3,4,7,8,find(isnan(mu))'];
    assert(all(isnan(H(idx_del))),'Significant cluster in area that was going to be removed.');
    idx_show = 1:size(H,1);
    idx_show(idx_del) = [];
    idx_show(end) = [];

    % individual datapoints
    for i_ROI = find(~isnan(mu))'
        % mn_sinaplot(subj_x_roi(:,i_ROI),[],find(idx_show == i_ROI),[],[],0.3);
        x = ones(size(subj_x_roi,1),1) * find(idx_show == i_ROI);
        scatter(x,subj_x_roi(:,i_ROI),'Jitter','on','Cdata',[0.8 0.8 0.8],...
                'MarkerFaceAlpha',0.2,'MarkerFaceColor','flat');
    end

    % barplot and SEM
    b = bar(mu(idx_show),'FaceColor','flat');
    b.FaceAlpha = 0.6;
    b.LineWidth = 1;
    b.EdgeColor = 'k';
    x = 1:numel(ROIs(idx_show));

    % change color based on valence
    colors = get_matlab_colors();
    idx = (mu(idx_show) > 0);
    b.CData(find(idx),:) = repmat(colors(2,:),sum(idx),1);
    b.CData(~idx,:) = repmat(colors(1,:),sum(~idx),1);

    % add CIs
    er = errorbar(x,mu(idx_show),mu(idx_show)-sem(1,idx_show)',sem(2,idx_show)'-mu(idx_show));   
    er.Color = [0 0 0];                            
    er.LineStyle = 'none';  
    er.LineWidth = 1;
    er.CapSize = er.LineWidth;

    lims = ylim;
    ylim([-max(abs(lims)),max(abs(lims))]);

    % make pretty
    a = get(gca,'XTickLabel');  
    set(gca,'XTickLabel',a,'fontsize',16,'FontWeight','normal'); % ,'FontWeight','bold'
    set(gca,'XTickLabelMode','auto')
    set(gca,'linewidth',1);

    % add labels
    xticks(1:numel(ROIs(idx_show)));
    xticklabels(ROI_names(idx_show));
    set(gca,'TickLabelInterpreter','none');
    xtickangle(45); 
    
    switch curr_pmod
        case 'SV_ch', curr_title = 'Subjective value';
        case 'APE_fb', curr_title = 'Action prediction error';
        case 'subj_KL_div_fb', curr_title = 'Belief updates';
    end
    title(curr_title,'Interpreter','none');
    
end
fprintf('\n');

end
