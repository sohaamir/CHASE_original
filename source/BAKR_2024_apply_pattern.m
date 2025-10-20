function [r,BU_hat,rmse] = BAKR_2024_apply_pattern(image_folder,project_folder,pattern)
%
% Predicting the extent of interpersonal belief update from neural activation
% patterns, based on a previsouly identified neural fingerprint of adaptive 
% mentalization (Bürgi, Aydogan, Konovalov, & Ruff; in prep).
%
% Inputs:
% - image_folder: path to folder containing mean images per bin and subject
%   named 'subj_[subjNo]_bin_[binNo]_mean.nii'.
% Outputs:
% - r: vector containing a correlation coefficient between actual and predicted
%   extent of belief update for each subject.
% - BU_hat: matrix containing the predicted labels for each subject and bin.
%

% find mean images for each bin for each subj
images = dir(fullfile(image_folder,'subj_*'));
subj = cellfun(@(x) str2double(x{2}), cellfun(@(x) strsplit(x,'_'),{images.name},'UniformOutput',0));
n_bins = mode(histcounts(subj,unique(subj)));
n_subj = max(subj);

% get grey matter mask and mentalization fingerprint (i.e. multi-variate pattern)
mask = fullfile(project_folder,'masks','gm.nii');
if nargin < 3
    pattern = fullfile(project_folder,'pattern','fingerprint.nii');
    intercept = -0.5920;
else
    warning('Using non-default pattern; ignoring intercept.');
    intercept = 0;
end

% apply to each subj and each bin to get predicted belief update (BU)
BU_hat = NaN(n_subj,n_bins);
for i_subj = 1:n_subj
    fprintf('\nSubj %d... ',i_subj);
    for i_bin = 1:n_bins
        img_file = fullfile(image_folder,sprintf('subj_%i_bin_%i_mean.nii',i_subj,i_bin));
        if exist(img_file,'file')
            dat = fmri_data(img_file,mask,'noverbose');
            BU_hat(i_subj,i_bin) = apply_mask(dat,pattern,'pattern_expression','ignore_missing') + intercept; % dot product plus intercept
        end
    end
end

% compute correlation and RMSE between predicted and actual belief update
r = NaN(n_subj,1);
rmse = NaN(n_subj,1);
for i_subj = 1:n_subj
	r(i_subj) = corr([1:n_bins]',BU_hat(i_subj,:)','rows','pairwise');
    rmse(i_subj) = sqrt(mean((BU_hat(i_subj,:) - linspace(-1,1,5)).^2));
end

end