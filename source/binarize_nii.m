function binarize_nii(inputfile,outputfile,valence)

if nargin < 1
    error('Please provide a path to a nifti file.');
end
if nargin < 2 || isempty(outputfile)
    fileparts = strsplit(inputfile,filesep);
    fileparts{end} = strcat(fileparts{end}(1:end-4),'_binarized.nii');
    outputfile = strjoin(fileparts,filesep);
end
if nargin < 3
    valence = 'both';
end

% inputfile = 'D:\2022-StressStrategic\weights_gm_thresh_fdr_05.nii';
nii = niftiread(inputfile);
info = niftiinfo(inputfile);

switch valence
    case 'pos', nii = (nii > 0);
    case 'neg', nii = (nii < 0);
    case 'both', nii = (nii ~= 0 & ~isnan(nii));
end

nii = eval(strcat(info.Datatype,'(nii)'));
niftiwrite(nii,outputfile,info);

end