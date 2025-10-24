%
% Results and figures for Buergi, Aydogan, Konovalov, & Ruff (2024):
% "A neural fingerprint of adaptive mentalization" 
%
% To re-create the files used herein, use run_model_fitting
%

%% set folder

clc; clearvars;
project_folder = cd;

% Define a separate output directory
output_dir = fullfile(project_folder, 'results', 'llm_subset');

% code
addpath(fullfile(project_folder,'source'));
addpath(fullfile(project_folder,'source','MERLIN_toolbox'));
addpath(genpath(fullfile(project_folder, 'VBA-toolbox')));

%% preparation

% plotting
colors = get_matlab_colors();
level_colors(1,:) = [50,130,150]/255;
level_colors(2,:) = [60,130,100]/255;
level_colors(3,:) = [130,170,80]/255;
level_colors(4,:) = [180,200,60]/255;

% set(groot,'DefaultAxesLineWidth',2)
% set(groot,'defaultAxesFontSize',12)

% data
results_folder = output_dir;
load(fullfile(results_folder,'fits_CHASE_table.mat'),'data'); % in table format
all_fits = data;

% === DIAGNOSTIC: Check data structure ===
fprintf('\n=== DATA STRUCTURE DIAGNOSTIC ===\n');
fprintf('all_fits type: %s\n', class(all_fits));
fprintf('all_fits size: %d rows × %d columns\n', height(all_fits), width(all_fits));
fprintf('all_fits columns: %s\n', strjoin(all_fits.Properties.VariableNames, ', '));
fprintf('Number of unique subjects: %d\n', numel(unique(all_fits.subjID)));
fprintf('First subject ID: %d\n', all_fits.subjID(1));
fprintf('Sample row (subject 1, trial 1):\n');
disp(all_fits(1,:));
fprintf('=== END DIAGNOSTIC ===\n\n');

%% -------------------------------------------------------------------------- %
%                    Fig 2: Behavioral findings
% -------------------------------------------------------------------------- %%

try
    fprintf('\n=== Starting Figure 2: Behavioral findings ===\n');
    
    figure('Name','Behavioral and model-based evidence for adaptive mentalization.',...
           'Units','normalized','Position',[0.2,0.2,0.6,0.6]);

    %% 2b) model comparison
    fprintf('  → Section 2b: Model comparison\n');

    % load the fits
    load(fullfile(results_folder,'model_comparison.mat'));

    fprintf('\n=== Checking fits structure ===\n');
    fprintf('Number of models: %d\n', numel(fits));
    for i = 1:numel(fits)
        fprintf('Model %d: %s\n', i, fits(i).model.name);
        fprintf('  Number of subjects: %d\n', numel(fits(i).subj));
        if numel(fits(i).subj) > 0
            fprintf('  Subject 1 fields: ');
            disp(fieldnames(fits(i).subj(1)));
            if isfield(fits(i).subj(1), 'optim')
                fprintf('  Subject 1 optim fields: ');
                disp(fieldnames(fits(i).subj(1).optim));
            else
                fprintf('  WARNING: optim field is missing!\n');
            end
        end
    end
    fprintf('=============================\n\n');

    subplot(3,9,4:6);
    [~,stats] = mn_compare(fits,'group','dataset','use_current_fig');
    set(gca,'FontSize',16);
    title('Model comparison');
    xlabel('Protected exceedance prob.');
    set(gca,'FontSize',12);
    l = legend;
    set(legend,'FontSize',8,'Position',[0.25,0.65,0.1,0.1]);

    yticklabels({'Fict','EWA','RL','ToMk','EWA-S','CHASE'});

    exportgraphics(gca, fullfile(output_dir, 'model_comparison.png'), 'Resolution', 300);

    pxp_dataset = arrayfun(@(dataset) dataset.rand.AIC.pxp(1),stats)

    % per opponent type (not applicable for LLM data)
    % [~,stats] = mn_compare(fits,'group','opp_type','flag_plot',0);
    % pxp_opp_type = arrayfun(@(dataset) dataset.rand.AIC.pxp(1),stats)

    %% 2c) model recovery
    fprintf('  → Section 2c: Model recovery\n');

    subplot(3,9,7:9);
    load(fullfile(results_folder,'model_recovery.mat'),'sim_fits');
    counts = BAKR_2024_model_recovery_plot(sim_fits,[1,6,3,5,2,4]);

    % shorten labels
    a = gca;
    a.YTickLabel{3} = 'EWA-S';
    a.YTickLabel{4} = 'Fict';
    a.YTickLabel{5} = 'RL';
    a.YTickLabel{6} = 'EWA';

    y = yticklabels;
    for i_y = 1:numel(y)
        y{i_y} = sprintf('%s (%i)',y{i_y},i_y);
    end
    yticklabels(y);
    xticklabels({'(1)','(2)','(3)','(4)','(5)','(6)'});
    xtickangle(0);

    % misattributions
    mean(counts(1,2:end))
    mean(counts(2:end,1))

    %% 2d) posterior predictive check
    fprintf('  → Section 2d: Posterior predictive check\n');

    BAKR_2024_posterior_predictive_check(results_folder,'main',{3,9,10:15});

    %% 2e) performance per opponent type (model-free)
    fprintf('  → Section 2e: Performance per opponent type\n');

    fit = all_fits(contains(all_fits.dataset,{'HUMAN','DEEPSEEK','GPT'}),:);
    subjects = unique(fit.subjID);

    % compute scores against different opponent types
    win_rate = NaN(numel(subjects),3);
    for i_subj = 1:numel(subjects)
        for bot = 0:2
            idx = (fit.subjID == subjects(i_subj) & fit.bot_level == bot);
            win_rate(i_subj,bot+1) = nanmean(fit.score_own(idx));
        end
    end

    % compute chance (whole sample)
    n = 1000;
    pi = [0 1 -1]; % <- overall score
    sample_score = NaN(n,1);
    for ii = 1:n
        a = mnrnd(80,[1/3 1/3 1/3],numel(subjects));
        individual_scores = a * pi' / 80; 
        sample_score(ii) = mean(individual_scores);
    end    
    chance_upper = prctile(sample_score,97.5);
    chance_lower = prctile(sample_score,2.5);

    % plot
    subplot(3,9,16:18);
    fill([0.26,0.26,3.75,3.75],[chance_lower,chance_upper,chance_upper,chance_lower],[0.9,0.9,0.9],'FaceAlpha',0.8,'EdgeAlpha',0);
    for ii = 1:3
        mn_sinaplot(win_rate(:,ii),-1:0.01:1,ii,level_colors(ii,:),20,0.12);
    end

    title('Overall score against different opponent types');
    ylim([-0.35,0.65]);
    yticks(-0.2:0.2:0.6);
    xlim([0.25,3.75]);
    xticks(1:3);
    xticklabels({"k=0","k=1","k=2"});
    xlabel("Opponent level");
    box off

    % ----------------------------------- stats ---------------------------------- %

    % test against chance per level
    for k = 1:3
        [~,p(k),~,STATS] = ttest(win_rate(:,k));
        tstat(k) = STATS.tstat;
    end
    max(p)
    min(tstat)

    % mixed model for effect of k
    t = table();
    t.subj = repmat(1:size(win_rate,1),1,3)';
    t.k = repelem(1:3,size(win_rate,1))';
    t.score = win_rate(:);
    lme = fitlme(t,'score ~ 1 + k + (k|subj)')
    tbl = anova(lme,'DFMethod','satterthwaite')

    % ---------------------------- model-based stats ----------------------------- %
    fprintf('  → Section 2e: Model-based stats\n');

    % kappa distribution
    subjects = unique(fit.subjID);
    for i_subj = 1:numel(subjects)
        kappa(i_subj) = unique(fit.kappa(fit.subjID == subjects(i_subj)));
    end
    histcounts(kappa,'Normalization','probability')

    % correctly inferring level-2 opponent above chance (belief > .5)
    correct_inf = NaN(numel(subjects),1);
    for i_subj = 1:numel(subjects)
        correct_inf(i_subj) = any(fit.beliefs(fit.subjID == subjects(i_subj) & fit.bot_level == 2,3) > 0.5);
    end
    mean(correct_inf)

    %% 2f) gameplay per opponent type (model-based)
    fprintf('  → Section 2f: Gameplay per opponent type\n');

    % infer played level based on best-response to beliefs (i.e. k+1)
    fit.exp_k_played = zeros(height(fit),4);
    fit.exp_k_played(fit.kappa == 0,1) = 1;
    fit.exp_k_played(fit.kappa == 1,2) = 1;
    fit.exp_k_played(fit.kappa == 2,2:3) = fit.beliefs(fit.kappa == 2,1:2);
    fit.exp_k_played(fit.kappa == 3,2:4) = fit.beliefs(fit.kappa == 3,:);

    subjects = unique(fit.subjID);
    n_subj = numel(subjects);

    for curr_bot = 0:2

        subplot(3,9,[19,20]+curr_bot*2); hold on;
        idx = (fit.bot_level == curr_bot);

        % count trials above cutoff
        level_counts = NaN(numel(subjects),4);
        for i_subj = 1:n_subj
            idx_subj = (idx & fit.subjID == subjects(i_subj));
            level_counts(i_subj,:) = mean(fit.exp_k_played(idx_subj,:) > 0.5); % exp_p_k
        end

        b = bar(mean(level_counts),'FaceAlpha',0.5,'FaceColor',level_colors(curr_bot+1,:),'EdgeColor',level_colors(curr_bot+1,:),'LineWidth',1.5);
        b = bar(sort(level_counts',2),'FaceAlpha',0.2,'FaceColor',level_colors(curr_bot+1,:),'EdgeColor',level_colors(curr_bot+1,:),'EdgeAlpha',0.2);

        xticks(1:4); xticklabels(0:3); 
        xlim([0,5]); ylim([0,1]);
        set(gca,'linewidth',1.5);
        yticks(0:0.2:1); yticklabels({});
        box off
        
        if curr_bot == 0
            yticklabels({'0','0.2','0.4','0.6','0.8','1'});
            ylabel("% of trials");
        end
        if curr_bot == 1
            title("Gameplay per opponent type");
            xlabel("Estimated subject level");
        end
        
    end

    %% 2g) opponent level belief updates
    fprintf('  → Section 2g: Belief updates\n');

    % extract z-scored timecourses
    fit_z = fit;
    kl_div = [];
    ii = 1;
    for subj = unique(fit.subjID)'
        fit_z.subj_KL_div(fit.subjID == subj & ~fit.missing) = zscore(fit.subj_KL_div(fit.subjID == subj & ~fit.missing));
        n_blocks_per_subj = numel(unique(fit.block));
        for block = 1:n_blocks_per_subj
            kl_div(ii,:) = fit_z.subj_KL_div(fit_z.subjID == subj & fit_z.block == block);
            ii = ii + 1;
        end
    end

    % plot
    subplot(3,9,25:27); hold on;
    p = plot(kl_div','Color',[colors(1,:) 0.02]);
    stdshade(kl_div,0.2,colors(1,:));
    plot(nanmean(kl_div), 'k', 'LineWidth',1);
    ylim([-1.5,4]);
    xlim([0,40]);
    xlabel("Trials with current opponent");
    % ylabel("Belief update");
    title("Opponent level belief updates");

    % ------------------------- additional analyses ------------------------------ &

    % time course
    data = table();
    kl_div_T = kl_div';
    n_blocks = 6;
    n_trials = 40;
    n_subj = size(kl_div,1)/n_blocks;
    data.kl_div = kl_div_T(:);
    data.trial = repmat([1:40]',n_subj*n_blocks,1);
    data.subj = repelem([1:n_subj]',n_blocks*n_trials,1);
    lme = fitlme(data,'kl_div ~ 1 + trial + (trial | subj)')
    tbl = anova(lme,'DFMethod','satterthwaite')

    fprintf('✓ Figure 2 complete\n\n');
    
catch e
    fprintf('\n❌ ERROR in Figure 2:\n');
    fprintf('Message: %s\n', e.message);
    fprintf('Location: %s (line %d)\n', e.stack(1).name, e.stack(1).line);
    fprintf('Full stack trace:\n');
    for i = 1:length(e.stack)
        fprintf('  [%d] %s (line %d)\n', i, e.stack(i).name, e.stack(i).line);
    end
    fprintf('Check fprintf output above for last successful section\n\n');
    rethrow(e);
end

%% -------------------------------------------------------------------------- %
%                          Supplementary analyses
% -------------------------------------------------------------------------- %%

%% level distribution across LLM and human data

fprintf('  → Level distribution analysis\n');

% Extract kappa values from fitted CHASE model
subjects = unique(all_fits.subjID);
kappa = NaN(numel(subjects), 1);
dataset_labels = cell(numel(subjects), 1);

for i_subj = 1:numel(subjects)
    idx = find(all_fits.subjID == subjects(i_subj), 1);
    kappa(i_subj) = all_fits.kappa(idx);
    dataset_labels{i_subj} = all_fits.dataset{idx};
end

% Create dataset groupings
idx_all = true(numel(kappa), 1);
idx_deepseek = contains(dataset_labels, 'DEEPSEEK');
idx_gpt = contains(dataset_labels, 'GPT');
idx_human = contains(dataset_labels, 'HUMAN');

% Verify counts
fprintf('    All data: %d subjects\n', sum(idx_all));
fprintf('    DeepSeek (all): %d subjects\n', sum(idx_deepseek));
fprintf('    GPT: %d subjects\n', sum(idx_gpt));
fprintf('    Human: %d subjects\n', sum(idx_human));

% Create figure
figure('Position', [100, 100, 1200, 400]);

% Panel 1: All data
subplot(1, 3, 1);
histogram(kappa(idx_all), 'BinEdges', -0.5:3.5, 'Normalization', 'probability', ...
          'FaceColor', [0.4, 0.6, 0.8], 'EdgeColor', 'k', 'LineWidth', 1);
ylim([0, 0.5]);
xlabel('Levels (κ)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Frequency', 'FontSize', 12, 'FontWeight', 'bold');
title('All Data', 'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'FontSize', 11, 'LineWidth', 1.5);
box off;

% Panel 2: DeepSeek (combined NORMAL + SCOT)
subplot(1, 3, 2);
histogram(kappa(idx_deepseek), 'BinEdges', -0.5:3.5, 'Normalization', 'probability', ...
          'FaceColor', [0.3, 0.7, 0.5], 'EdgeColor', 'k', 'LineWidth', 1);
ylim([0, 0.5]);
xlabel('Levels (κ)', 'FontSize', 12, 'FontWeight', 'bold');
title('DeepSeek (All)', 'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'FontSize', 11, 'LineWidth', 1.5);
box off;

% Panel 3: GPT
subplot(1, 3, 3);
histogram(kappa(idx_gpt), 'BinEdges', -0.5:3.5, 'Normalization', 'probability', ...
          'FaceColor', [0.8, 0.5, 0.3], 'EdgeColor', 'k', 'LineWidth', 1);
ylim([0, 0.5]);
xlabel('Levels (κ)', 'FontSize', 12, 'FontWeight', 'bold');
title('GPT', 'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'FontSize', 11, 'LineWidth', 1.5);
box off;

% Overall title
sgtitle('Level Distribution by Model Architecture', 'FontSize', 16, 'FontWeight', 'bold');

% Save figure
exportgraphics(gcf, fullfile(output_dir, 'level_distribution_by_architecture.png'), 'Resolution', 300);
fprintf('    ✓ Saved: level_distribution_by_architecture.png\n');

% Summary statistics
fprintf('\n    Level distribution summary:\n');
fprintf('      All: κ = %.2f ± %.2f (mean ± SD)\n', mean(kappa(idx_all)), std(kappa(idx_all)));
fprintf('      DeepSeek: κ = %.2f ± %.2f\n', mean(kappa(idx_deepseek)), std(kappa(idx_deepseek)));
fprintf('      GPT: κ = %.2f ± %.2f\n', mean(kappa(idx_gpt)), std(kappa(idx_gpt)));
fprintf('      Human: κ = %.2f ± %.2f\n', mean(kappa(idx_human)), std(kappa(idx_human)));

% Optional: Breakdown by DeepSeek variants
idx_ds_normal = strcmp(dataset_labels, 'DEEPSEEK-NORMAL');
idx_ds_scot = strcmp(dataset_labels, 'DEEPSEEK-SCOT');
fprintf('      DeepSeek-Normal: κ = %.2f ± %.2f\n', mean(kappa(idx_ds_normal)), std(kappa(idx_ds_normal)));
fprintf('      DeepSeek-SCOT: κ = %.2f ± %.2f\n', mean(kappa(idx_ds_scot)), std(kappa(idx_ds_scot)));

fprintf('\n');

%% LR recovery

figure; 
load(fullfile(results_folder,'supplementary','model_recovery_LR.mat'),'sim_fits');  % ✓ FIX
BAKR_2024_model_recovery_plot(sim_fits,[1,2,4,3]);

% exportgraphics(gcf,'model_recovery_LR.png','Resolution',300);

%% LR comparisons - BY DATASET

fprintf('  → Learning rule model comparison by dataset\n');

load(fullfile(output_dir,'supplementary','model_comparison_LR.mat'),'fits_LR');
fits_LR(3:4) = [];  % Remove unused models

% Model comparison BY DATASET
figure('Position', [100, 100, 800, 500]);

% Single panel: Individual datasets
[~, stats_datasets] = mn_compare(fits_LR, 'group', 'dataset', 'use_current_fig');
title('Learning Rule Comparison by Dataset', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Protected Exceedance Probability (PXP)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Learning Rule', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'FontSize', 11, 'LineWidth', 1.5);

% Adjust legend
legend_handle = findobj(gcf, 'Type', 'Legend');
if ~isempty(legend_handle)
    set(legend_handle, 'FontSize', 10, 'Location', 'best');
end

% Save figure
exportgraphics(gcf, fullfile(output_dir, 'learning_rule_comparison_by_dataset.png'), 'Resolution', 300);
fprintf('    ✓ Saved: learning_rule_comparison_by_dataset.png\n');

% Extract PXP values and dataset names
pxp_dataset = arrayfun(@(dataset) dataset.rand.AIC.pxp(1), stats_datasets);

% FIXED: Extract dataset names from the fitted data
datasets_raw = arrayfun(@(subj) subj.data.dataset, fits_LR(1).subj, 'UniformOutput', false);
datasets = unique(vertcat(datasets_raw{:}));  % Flatten nested cells

fprintf('\n    PXP values by dataset (for best learning rule):\n');
for i = 1:numel(datasets)
    fprintf('      %s: PXP = %.3f\n', datasets{i}, pxp_dataset(i));
end

% Additional statistics: Compare learning rules within each dataset
fprintf('\n    Within-dataset learning rule preferences:\n');
for i = 1:numel(stats_datasets)
    pxp_all = stats_datasets(i).rand.AIC.pxp;
    [~, best_idx] = max(pxp_all);
    
    % Get model names from fits_LR
    model_names = arrayfun(@(f) f.model.name, fits_LR, 'UniformOutput', false);
    
    fprintf('      %s:\n', datasets{i});
    fprintf('        Best: %s (PXP = %.3f)\n', model_names{best_idx}, pxp_all(best_idx));
    
    % Show all PXPs for this dataset
    for j = 1:numel(pxp_all)
        if j ~= best_idx
            fprintf('        %s: PXP = %.3f\n', model_names{j}, pxp_all(j));
        end
    end
end

fprintf('\n');

%% param recovery

load(fullfile(results_folder,'supplementary','parameter_recovery.mat'),'prec');  % ✓ FIX

figure;
params = {prec.model.params.name};
gen = prec.params.gen;
est = prec.params.est;
est(est(:,end) < 2,3) = NaN; % no gamma
est(est(:,end) == 0,2) = NaN; % no lambda, either
param_lims = [0.4,4.2; 0,3.6; 0,10; 0,1; -0.5,3.5];

% individual parameters
ii = 1;
for i_param = [4,1:3,5]
    subplot(2,numel(params),ii); hold on;
    plot(param_lims(i_param,:),param_lims(i_param,:),'k--','Color',ones(1,4)*0.7);
    if i_param == 5
        noise = normrnd(0,0.15,size(gen,1),2);
        scatter(gen(:,i_param)+noise(:,1),est(:,i_param)+noise(:,2),'filled','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.3,'MarkerEdgeColor',colors(1,:),'CData',colors(1,:));
    else
        scatter(gen(:,i_param),est(:,i_param),'filled','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.3,'MarkerEdgeColor',colors(1,:),'CData',colors(1,:));
    end
    [r,p] = corr(gen(:,i_param),est(:,i_param),'rows','pairwise');
    title({params{i_param},sprintf('r = %.2f',r)});
    xlim(param_lims(i_param,:)), ylim(param_lims(i_param,:));
    xlabel('Generating');
    axis square;
    ii = ii + 1;
    if i_param == 3, rectangle('Position',[0,0,2.5,2.5],'LineWidth',0.1,'EdgeColor',[0 0 0 0.7]); end
    if i_param == 4, ylabel('Recovered'); end
end

% zoom in on most relevant gamma range
subplot(2,numel(params),9); hold on;
lims = [0,2.5];
plot(lims,lims,'k--','Color',ones(1,4)*0.7);
scatter(gen(:,3),est(:,3),'filled','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.3,'MarkerEdgeColor',colors(1,:),'CData',colors(1,:));
rectangle('Position',[0,0,2.5,2.5],'LineWidth',0.1,'EdgeColor',[0 0 0 0.7]);
xlim(lims); ylim(lims);
xlabel('Generating');
axis square;
box off;

% exportgraphics(gcf,'param_rec.png','Resolution',300);

mean(gen(:,3) <= 2.5)

%% effect of lowering the upper bound on gamma

load(fullfile(results_folder,'supplementary','fits_bounded.mat'),'fits_bounded');

gamma = arrayfun(@(subj) subj.params.gamma,fits_bounded(1).subj);
mean(gamma > 10)
gamma = arrayfun(@(subj) subj.params.gamma,fits_bounded(2).subj);
mean(gamma == 10)
mean(gamma > 2.5)

idx_high_gamma = find(gamma > 2.5);

clear negLL
for i_fit = 1:numel(fits_bounded)
    negLL(:,i_fit) = arrayfun(@(subj) subj.optim.negLL,fits_bounded(i_fit).subj(idx_high_gamma));
end
perc_decrease = max(negLL(:,2:end) - negLL(:,1),0)./negLL(:,1)*100;

% effect on likelihoods
figure; hold on;
yline(0,'k--');
plot(perc_decrease','Color',ones(1,4)*0.2);
scatter(repelem([1:(size(negLL,2)-1)]',size(negLL,1),1),perc_decrease(:),'CData',colors(1,:));
err = nanstd(perc_decrease)./sqrt(size(perc_decrease,1));
errorbar(nanmean(perc_decrease),err,':','Color',[0.1 0.1 0.1 0.8],'LineWidth',2)
scatter(1:size(perc_decrease,2),nanmean(perc_decrease),'SizeData',75,'MarkerFaceColor',colors(1,:),'MarkerEdgeColor','k','LineWidth',2);
xticks(1:numel(fits_bounded)); xticklabels([10,5,2.5,1,0.5,0.1]);
xlabel('Upper bound on gamma');
ylabel({'% decrease in likelihood','(relative to an unbounded model)'});
lims = ylim();
ylim([-0.25,lims(2)]);
xlim([0.5,size(perc_decrease,2)+0.5]);
f = gcf;
f.Position = [1000 1022 493 216];

% exportgraphics(gcf,'gamma_UB_likelihoods.png','Resolution',300)

% effect on BU
ii = 1;
figure; hold on;
for i_fit = [2,4]

    BU = arrayfun(@(subj) subj.states.subj_KL_div,fits_bounded(i_fit).subj,'UniformOutput',0);

    BU = cat(2,BU{:});
    for i_subj = 1:size(BU,2)
        idx = ~isnan(BU(:,i_subj));
        BU(idx,i_subj) = zscore(log(BU(idx,i_subj)+1e-3));
    end
    l(ii) = stdshade(BU',0.4,colors(ii,:),[],[],'sem',0.7);
    BUs(:,:,ii) = BU;
    UB(ii) = fits_bounded(i_fit).model.params(3).support(2);
    ii = ii + 1;
    
end
xlim([0,240]);
xlabel('Trial');
xline([41,81,121,161,201],'--','Color',ones(1,3)*0.6,'LineWidth',2);
legend(l,{['UB = ' num2str(UB(1))],['UB = ' num2str(UB(2))]});
ylabel('Belief update');

f = gcf;
f.Position = [770 1064 721 142];
% exportgraphics(gcf,'gamma_UB_BUs.png','Resolution',300)

% correlations
for i_subj = 1:size(BUs,2)
    r(i_subj) = corr(BUs(:,i_subj,1),BUs(:,i_subj,2),'rows','pairwise');
end
tanh(mean(atanh(r(idx_high_gamma))))
mean(r>.8)

%% posterior predictive for alternative models

BAKR_2024_posterior_predictive_check(results_folder,'all')

%% parameter correlations

fprintf('  → Parameter correlations analysis\n');

% extract estimates
data_subjlevel = table();
subjects = unique(all_fits.subjID);
n_subj = numel(subjects);
vars = {'subjID','dataset','alpha','beta','gamma','lambda','kappa'};
for i_subj = 1:n_subj
    idx_first = find(all_fits.subjID == subjects(i_subj),1);
    new_data = table();
    for i_var = 1:numel(vars)
        new_data.(vars{i_var}) = all_fits.(vars{i_var})(idx_first);
    end
    data_subjlevel = [data_subjlevel; new_data];
end

% parameter censoring (as certain parameters are effectively removed from
% the model for low values of kappa)
data_subjlevel.gamma(data_subjlevel.kappa < 2) = NaN;
data_subjlevel.lambda(data_subjlevel.kappa == 0) = NaN;

params = vars;
data = data_subjlevel;
datasets = unique(data.dataset);

fprintf('  Found %d unique datasets: %s\n', numel(datasets), strjoin(datasets, ', '));

% correlations
[R_all,P_all] = corr(table2array(data(:,3:7)),'type','Spearman','rows','pairwise');
R_all .* double(P_all < 0.01)

figure;
subplot(3,6,[1:3,7:9,13:15]);
imagesc(R_all,[-1,1]); cb = colorbar;
yl = ylabel(cb,'Correlation','FontSize',12,'Rotation',270,'FontWeight','bold');
idx_fig = [4:6,10:12,16:18];
title('Pooled data');
axis square

xticks(1:numel(params)); xticklabels(params);
yticks(1:numel(params)); yticklabels(params);

clear new_Rs Rs
params = {'alpha','beta','gamma','lambda','kappa'};
for i_dataset = 1:numel(datasets)
    idx = strcmp(data.dataset,datasets{i_dataset});
    [R,P] = corr(table2array(data(idx,3:7)),'type','Spearman','rows','pairwise');
    subplot(3,6,idx_fig(i_dataset));
    imagesc(R,[-1,1]);
    hold on;
    
    % Better title formatting for LLM datasets
    dataset_short = datasets{i_dataset};
    if contains(dataset_short, 'DEEPSEEK')
        if contains(dataset_short, 'SCOT')
            dataset_short = 'DS-CoT';
        else
            dataset_short = 'DS';
        end
    elseif contains(dataset_short, 'GPT')
        dataset_short = 'GPT';
    elseif contains(dataset_short, 'HUMAN')
        dataset_short = 'HUM';
    end
    title(dataset_short);
    
    xticks(1:numel(params)); xticklabels({''});
    yticks(1:numel(params)); yticklabels({''});
    axis square
    
    new_Rs = R(tril(ones(numel(params)),-1) == 1);
    Rs(:,i_dataset) = new_Rs;
end

% exportgraphics(gcf,'parameter_correlations.png');

r_pos = tril(reshape([1:25],5,5)',-1);
r_pos = r_pos(r_pos ~= 0);

%% effect of dataset features on parameter correlations - ADAPTED FOR LLM DATA
fprintf('  → Analyzing parameter correlations by dataset features\n');

t = table();
t.dataset = datasets;

% Define contrasts relevant to LLM data (FIX: Remove transpose operator)
t.human = double(contains(datasets, 'HUMAN'));      % Human vs LLM (0=LLM, 1=Human)
t.deepseek = double(contains(datasets, 'DEEPSEEK')); % DeepSeek vs GPT (1=DeepSeek, 0=GPT)
t.scot = double(contains(datasets, 'SCOT'));         % Chain-of-thought vs normal (1=SCOT, 0=Normal)

fprintf('\n  Dataset feature coding:\n');
fprintf('    %-20s Human  DeepSeek  SCOT\n', 'Dataset');
fprintf('    %s\n', repmat('-', 1, 45));
for i = 1:numel(datasets)
    fprintf('    %-20s   %d       %d       %d\n', ...
        datasets{i}, t.human(i), t.deepseek(i), t.scot(i));
end
fprintf('\n');

% DIAGNOSTIC: Check dimensions
fprintf('\n=== DIMENSION DEBUG ===\n');
fprintf('datasets class: %s\n', class(datasets));
fprintf('datasets size: %s\n', mat2str(size(datasets)));

fprintf('\nTable height after t.dataset = datasets: %d\n', height(t));
fprintf('Expected column size: [%d, 1]\n', height(t));
fprintf('======================\n\n');

% Test each parameter correlation
significant_results = [];
for i_corr = 1:size(Rs,1)
    t.r = Rs(i_corr,:)';
    
    % Test each predictor separately (since N=4 is small for multiple regression)
    lm_human = fitlm(t, 'r ~ 1 + human');
    lm_deepseek = fitlm(t, 'r ~ 1 + deepseek');
    lm_scot = fitlm(t, 'r ~ 1 + scot');
    
    % Check for significance (Bonferroni corrected across all correlations and predictors)
    alpha_corrected = 0.05 / (size(Rs,1) * 3);  % 3 predictors tested
    
    % Get parameter pair names
    param1_idx = mod(r_pos(i_corr)-1, 5) + 1;
    param2_idx = ceil(r_pos(i_corr) / 5);
    param_pair = sprintf('%s <-> %s', params{param1_idx}, params{param2_idx});
    
    % Test human effect
    if lm_human.Coefficients.pValue(2) < alpha_corrected
        fprintf('  *** SIGNIFICANT: HUMAN effect on %s correlation ***\n', param_pair);
        fprintf('      β = %.3f, p = %.4f\n', ...
            lm_human.Coefficients.Estimate(2), ...
            lm_human.Coefficients.pValue(2));
        fprintf('      Interpretation: Human vs LLM shows different correlation pattern\n\n');
        significant_results = [significant_results; ...
            {param_pair, 'Human', lm_human.Coefficients.Estimate(2), lm_human.Coefficients.pValue(2)}];
    end
    
    % Test DeepSeek effect
    if lm_deepseek.Coefficients.pValue(2) < alpha_corrected
        fprintf('  *** SIGNIFICANT: DEEPSEEK effect on %s correlation ***\n', param_pair);
        fprintf('      β = %.3f, p = %.4f\n', ...
            lm_deepseek.Coefficients.Estimate(2), ...
            lm_deepseek.Coefficients.pValue(2));
        fprintf('      Interpretation: DeepSeek vs GPT shows different correlation pattern\n\n');
        significant_results = [significant_results; ...
            {param_pair, 'DeepSeek', lm_deepseek.Coefficients.Estimate(2), lm_deepseek.Coefficients.pValue(2)}];
    end
    
    % Test SCOT effect
    if lm_scot.Coefficients.pValue(2) < alpha_corrected
        fprintf('  *** SIGNIFICANT: SCOT effect on %s correlation ***\n', param_pair);
        fprintf('      β = %.3f, p = %.4f\n', ...
            lm_scot.Coefficients.Estimate(2), ...
            lm_scot.Coefficients.pValue(2));
        fprintf('      Interpretation: Chain-of-thought vs normal prompting shows different correlation pattern\n\n');
        significant_results = [significant_results; ...
            {param_pair, 'SCOT', lm_scot.Coefficients.Estimate(2), lm_scot.Coefficients.pValue(2)}];
    end
end

% Summary
if isempty(significant_results)
    fprintf('  No significant effects found (α = %.4f, Bonferroni corrected)\n', alpha_corrected);
    fprintf('  This suggests parameter correlations are similar across:\n');
    fprintf('    - Humans vs LLMs\n');
    fprintf('    - DeepSeek vs GPT architectures\n');
    fprintf('    - Normal vs chain-of-thought prompting\n\n');
else
    fprintf('  SUMMARY: Found %d significant effect(s)\n', size(significant_results, 1));
    fprintf('  See detailed results above for interpretation\n\n');
    
    % Optional: save results to table
    results_table = cell2table(significant_results, ...
        'VariableNames', {'ParameterPair', 'Feature', 'Beta', 'pValue'});
    disp(results_table);
end

fprintf('  ✓ Parameter correlations analysis complete\n\n');


%% ========================================================================== %
%                    DATASET-SPECIFIC ANALYSES                               %
% ========================================================================== %%

fprintf('\n========================================\n');
fprintf('DATASET-SPECIFIC ANALYSES\n');
fprintf('========================================\n\n');

% Get datasets
datasets = unique(all_fits.dataset);
n_datasets = numel(datasets);

fprintf('Analyzing %d datasets:\n', n_datasets);
for i = 1:n_datasets
    fprintf('  %d. %s\n', i, datasets{i});
end
fprintf('\n');

% Define colors for each dataset
dataset_colors = struct();
dataset_colors.DEEPSEEK_NORMAL = [0.2, 0.4, 0.7];  % Blue
dataset_colors.DEEPSEEK_SCOT = [0.3, 0.7, 0.5];    % Green
dataset_colors.GPT_NORMAL = [0.9, 0.5, 0.2];       % Orange
dataset_colors.HUMAN = [0.7, 0.3, 0.5];            % Purple

%% -------------------------------------------------------------------------- %
%                    1. POSTERIOR PREDICTIVE CHECK BY DATASET                %
% -------------------------------------------------------------------------- %%

fprintf('1. Posterior predictive check by dataset\n');

try
    % Load simulations
    load(fullfile(results_folder,'simulations.mat'), 'sims', 'new_T');
    
    % Get CHASE simulations only
    chase_model_name = fits(1).model.name;  % Should be 'CHASE'
    sims_chase = sims(strcmp({sims.model}, chase_model_name));
    
    fprintf('   Found %d CHASE simulations\n', numel(sims_chase));
    
    % Create one figure per dataset
    for i_dataset = 1:n_datasets
        curr_dataset = datasets{i_dataset};
        fprintf('   Processing %s... ', curr_dataset);
        
        % Get data for this dataset
        data_real = new_T(strcmp(new_T.dataset, curr_dataset), :);
        
        % Get simulations for this dataset
        % Match by subjID (sims have subjID = model*100 + original_subj_idx)
        subj_ids_dataset = unique(data_real.subjID);
        
        % Find corresponding simulations
        % Note: simulations have different subjIDs, need to map by position
        idx_dataset_subjs = find(ismember([fits(1).subj.subjID], subj_ids_dataset));
        
        % Simulations for these subjects start at 100 + idx
        sim_ids = 100 + idx_dataset_subjs;
        sims_dataset = sims_chase(ismember([sims_chase.subjID], sim_ids));
        
        if isempty(sims_dataset)
            fprintf('no simulations found, skipping\n');
            continue;
        end
        
        % Create simulation table
        sim_table = table();
        for i_sim = 1:numel(sims_dataset)
            n_trials = sims_dataset(i_sim).n_trials;
            temp = table();
            temp.subjID = repmat(sims_dataset(i_sim).subjID, n_trials, 1);
            temp.trial = sims_dataset(i_sim).trial;
            temp.choice_own = sims_dataset(i_sim).choice_own;
            temp.choice_other = sims_dataset(i_sim).choice_other;
            temp.bot_level = repelem(sims_dataset(i_sim).bot_level, 40);
            sim_table = [sim_table; temp];
        end
        
        % Compute behavioral signatures
        [meanBR_real, SEBR_real, meanBR_sim, SEBR_sim] = ...
            compute_behavioral_signatures(data_real, sim_table);
        
        % Create figure
        fig = figure('Position', [100, 100, 1400, 400]);
        dataset_color = dataset_colors.(strrep(curr_dataset, '-', '_'));
        
        for bot_level = 0:2
            subplot(1, 3, bot_level + 1);
            hold on;
            
            % Plot simulated data (signature k+1 and other k)
            t = 1:40;
            
            % Signature of k+1 (correct level)
            sim_mean_correct = squeeze(meanBR_sim(bot_level + 1, bot_level + 1, :));
            sim_se_correct = squeeze(SEBR_sim(bot_level + 1, bot_level + 1, :));
            fill([t fliplr(t)], ...
                 [(sim_mean_correct + sim_se_correct)', ...
                  fliplr((sim_mean_correct - sim_se_correct)')], ...
                 dataset_color, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
            plot(t, sim_mean_correct, 'Color', dataset_color, 'LineWidth', 2);
            
            % Signature of other k (pooled)
            other_levels = setdiff(1:3, bot_level + 1);
            sim_mean_other = squeeze(mean(meanBR_sim(other_levels, bot_level + 1, :), 1));
            sim_se_other = squeeze(mean(SEBR_sim(other_levels, bot_level + 1, :), 1));
            fill([t fliplr(t)], ...
                 [(sim_mean_other + sim_se_other)', ...
                  fliplr((sim_mean_other - sim_se_other)')], ...
                 [0.5 0.5 0.5], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
            plot(t, sim_mean_other, 'Color', [0.5 0.5 0.5], 'LineWidth', 2);
            
            % Plot real data
            real_mean_correct = squeeze(meanBR_real(bot_level + 1, bot_level + 1, :));
            real_se_correct = squeeze(SEBR_real(bot_level + 1, bot_level + 1, :));
            errorbar(t, real_mean_correct, real_se_correct, ...
                     'o', 'Color', dataset_color, 'MarkerFaceColor', dataset_color, ...
                     'MarkerSize', 3, 'LineWidth', 0.75, 'CapSize', 2);
            
            real_mean_other = squeeze(mean(meanBR_real(other_levels, bot_level + 1, :), 1));
            real_se_other = squeeze(mean(SEBR_real(other_levels, bot_level + 1, :), 1));
            errorbar(t, real_mean_other, real_se_other, ...
                     'o', 'Color', [0.3 0.3 0.3], 'MarkerFaceColor', [0.5 0.5 0.5], ...
                     'MarkerSize', 3, 'LineWidth', 0.75, 'CapSize', 2);
            
            % Format
            xlabel('Trial', 'FontSize', 12);
            if bot_level == 0
                ylabel('Frequency relative to chance', 'FontSize', 12);
            end
            title(sprintf('Bot k = %d', bot_level), 'FontSize', 13, 'FontWeight', 'bold');
            ylim([-0.5, 0.6]);
            xlim([0, 41]);
            grid off;
            box off;
            
            % Legend on first panel
            if bot_level == 0
                legend({'Signature k+1 (sim)', '', 'Other k (sim)', '', ...
                        'Data k+1', 'Data other'}, ...
                       'Location', 'northwest', 'FontSize', 9);
            end
        end
        
        % Overall title
        sgtitle(sprintf('Posterior Predictive Check: %s', strrep(curr_dataset, '-', ' ')), ...
                'FontSize', 15, 'FontWeight', 'bold');
        
        % Save
        filename = sprintf('posterior_predictive_%s.png', strrep(curr_dataset, '-', '_'));
        exportgraphics(fig, fullfile(results_folder, filename), 'Resolution', 300);
        fprintf('saved\n');
        close(fig);
    end
    
    fprintf('   ✓ Posterior predictive checks complete\n\n');
    
catch e
    fprintf('\n   ❌ Error: %s\n', e.message);
    for i = 1:min(3, length(e.stack))
        fprintf('      [%d] %s (line %d)\n', i, e.stack(i).name, e.stack(i).line);
    end
    fprintf('\n');
end

%% -------------------------------------------------------------------------- %
%                    2. OVERALL SCORE BY DATASET                             %
% -------------------------------------------------------------------------- %%

fprintf('2. Overall score against opponent types by dataset\n');

try
    fig = figure('Position', [100, 100, 1400, 400]);
    
    for i_dataset = 1:n_datasets
        curr_dataset = datasets{i_dataset};
        fprintf('   Processing %s... ', curr_dataset);
        
        % Get data
        fit = all_fits(strcmp(all_fits.dataset, curr_dataset), :);
        subjects = unique(fit.subjID);
        
        % Compute scores per bot level
        win_rate = NaN(numel(subjects), 3);
        for i_subj = 1:numel(subjects)
            for bot = 0:2
                idx = (fit.subjID == subjects(i_subj) & fit.bot_level == bot);
                win_rate(i_subj, bot + 1) = nanmean(fit.score_own(idx));
            end
        end
        
        % Compute chance
        n = 1000;
        pi = [0 1 -1];
        sample_score = NaN(n, 1);
        for ii = 1:n
            a = mnrnd(80, [1/3 1/3 1/3], numel(subjects));
            individual_scores = a * pi' / 80;
            sample_score(ii) = mean(individual_scores);
        end
        chance_upper = prctile(sample_score, 97.5);
        chance_lower = prctile(sample_score, 2.5);
        
        % Plot
        subplot(1, n_datasets, i_dataset);
        hold on;
        
        dataset_color = dataset_colors.(strrep(curr_dataset, '-', '_'));
        
        % Chance band
        fill([0.26, 0.26, 3.75, 3.75], ...
             [chance_lower, chance_upper, chance_upper, chance_lower], ...
             [0.9, 0.9, 0.9], 'FaceAlpha', 0.8, 'EdgeAlpha', 0);
        
        % Violin plots
        for ii = 1:3
            mn_sinaplot(win_rate(:, ii), -1:0.01:1, ii, dataset_color, 20, 0.12);
        end
        
        % Format
        title(strrep(curr_dataset, '-', ' '), 'FontSize', 12, 'FontWeight', 'bold');
        ylim([-0.35, 0.65]);
        yticks(-0.2:0.2:0.6);
        xlim([0.25, 3.75]);
        xticks(1:3);
        xticklabels({"k=0", "k=1", "k=2"});
        xlabel("Opponent level", 'FontSize', 11);
        
        if i_dataset == 1
            ylabel("Overall score", 'FontSize', 11);
        end
        
        box off;
        
        fprintf('done\n');
    end
    
    sgtitle('Overall Score Against Different Opponent Types', ...
            'FontSize', 15, 'FontWeight', 'bold');
    
    exportgraphics(fig, fullfile(results_folder, 'overall_score_by_dataset.png'), 'Resolution', 300);
    fprintf('   ✓ Overall score analysis complete\n\n');
    close(fig);
    
catch e
    fprintf('\n   ❌ Error: %s\n', e.message);
    fprintf('\n');
end

%% -------------------------------------------------------------------------- %
%                    3. GAMEPLAY PER OPPONENT TYPE BY DATASET                %
% -------------------------------------------------------------------------- %%

fprintf('3. Gameplay per opponent type by dataset\n');

try
    % Compute expected level played
    all_fits.exp_k_played = zeros(height(all_fits), 4);
    all_fits.exp_k_played(all_fits.kappa == 0, 1) = 1;
    all_fits.exp_k_played(all_fits.kappa == 1, 2) = 1;
    
    idx_k2 = (all_fits.kappa == 2);
    all_fits.exp_k_played(idx_k2, 2:3) = all_fits.beliefs(idx_k2, 1:2);
    
    idx_k3 = (all_fits.kappa == 3);
    all_fits.exp_k_played(idx_k3, 2:4) = all_fits.beliefs(idx_k3, :);
    
    % Create figure
    fig = figure('Position', [100, 100, 1400, 900]);
    
    for i_dataset = 1:n_datasets
        curr_dataset = datasets{i_dataset};
        fprintf('   Processing %s... ', curr_dataset);
        
        fit = all_fits(strcmp(all_fits.dataset, curr_dataset), :);
        subjects = unique(fit.subjID);
        n_subj = numel(subjects);
        
        dataset_color = dataset_colors.(strrep(curr_dataset, '-', '_'));
        
        for curr_bot = 0:2
            subplot_idx = (i_dataset - 1) * 3 + curr_bot + 1;
            subplot(n_datasets, 3, subplot_idx);
            hold on;
            
            idx = (fit.bot_level == curr_bot);
            
            % Count trials above cutoff
            level_counts = NaN(n_subj, 4);
            for i_subj = 1:n_subj
                idx_subj = (idx & fit.subjID == subjects(i_subj));
                level_counts(i_subj, :) = mean(fit.exp_k_played(idx_subj, :) > 0.5);
            end
            
            % Plot
            b = bar(mean(level_counts), 'FaceAlpha', 0.5, 'FaceColor', dataset_color, ...
                    'EdgeColor', dataset_color, 'LineWidth', 1.5);
            bar(sort(level_counts', 2), 'FaceAlpha', 0.2, 'FaceColor', dataset_color, ...
                'EdgeColor', dataset_color, 'EdgeAlpha', 0.2);
            
            % Format
            xticks(1:4);
            xticklabels(0:3);
            xlim([0, 5]);
            ylim([0, 1]);
            yticks(0:0.2:1);
            
            if curr_bot == 0
                ylabel(strrep(curr_dataset, '-', ' '), 'FontSize', 11, 'FontWeight', 'bold');
            else
                yticklabels({});
            end
            
            if i_dataset == 1
                title(sprintf('vs Bot k=%d', curr_bot), 'FontSize', 11);
            end
            
            if i_dataset == n_datasets && curr_bot == 1
                xlabel("Estimated subject level", 'FontSize', 11);
            end
            
            box off;
        end
        
        fprintf('done\n');
    end
    
    sgtitle('Gameplay Per Opponent Type', 'FontSize', 15, 'FontWeight', 'bold');
    
    exportgraphics(fig, fullfile(results_folder, 'gameplay_by_dataset.png'), 'Resolution', 300);
    fprintf('   ✓ Gameplay analysis complete\n\n');
    close(fig);
    
catch e
    fprintf('\n   ❌ Error: %s\n', e.message);
    fprintf('\n');
end

%% -------------------------------------------------------------------------- %
%                    4. BELIEF UPDATES BY DATASET                            %
% -------------------------------------------------------------------------- %%

fprintf('4. Opponent level belief updates by dataset\n');

try
    fig = figure('Position', [100, 100, 1400, 400]);
    
    for i_dataset = 1:n_datasets
        curr_dataset = datasets{i_dataset};
        fprintf('   Processing %s... ', curr_dataset);
        
        fit = all_fits(strcmp(all_fits.dataset, curr_dataset), :);
        
        % Extract z-scored timecourses
        fit_z = fit;
        kl_div = [];
        ii = 1;
        
        for subj = unique(fit.subjID)'
            fit_z.subj_KL_div(fit.subjID == subj & ~fit.missing) = ...
                zscore(fit.subj_KL_div(fit.subjID == subj & ~fit.missing));
            
            n_blocks_per_subj = numel(unique(fit.block));
            for block = 1:n_blocks_per_subj
                kl_div(ii, :) = fit_z.subj_KL_div(fit_z.subjID == subj & fit_z.block == block);
                ii = ii + 1;
            end
        end
        
        % Plot
        subplot(1, n_datasets, i_dataset);
        hold on;
        
        dataset_color = dataset_colors.(strrep(curr_dataset, '-', '_'));
        
        % Individual traces
        plot(kl_div', 'Color', [dataset_color 0.02]);
        
        % Mean with shading
        stdshade(kl_div, 0.2, dataset_color);
        plot(nanmean(kl_div), 'Color', 'k', 'LineWidth', 2);
        
        % Format
        title(strrep(curr_dataset, '-', ' '), 'FontSize', 12, 'FontWeight', 'bold');
        ylim([-1.5, 4]);
        xlim([0, 40]);
        xlabel("Trial", 'FontSize', 11);
        
        if i_dataset == 1
            ylabel("Belief update (z-scored)", 'FontSize', 11);
        end
        
        box off;
        
        fprintf('done\n');
    end
    
    sgtitle('Opponent Level Belief Updates', 'FontSize', 15, 'FontWeight', 'bold');
    
    exportgraphics(fig, fullfile(results_folder, 'belief_updates_by_dataset.png'), 'Resolution', 300);
    fprintf('   ✓ Belief updates analysis complete\n\n');
    close(fig);
    
catch e
    fprintf('\n   ❌ Error: %s\n', e.message);
    fprintf('\n');
end

%% -------------------------------------------------------------------------- %
%                    5. PARAMETER RECOVERY BY DATASET                        %
% -------------------------------------------------------------------------- %%

fprintf('5. Parameter recovery by dataset\n');

try
    load(fullfile(results_folder,'supplementary','parameter_recovery_by_dataset.mat'), 'prec_by_dataset');
    
    dataset_fields = fieldnames(prec_by_dataset);
    
    fig = figure('Position', [100, 100, 1400, 350 * n_datasets]);
    
    params = {prec_by_dataset.(dataset_fields{1}).model.params.name};
    param_lims = [0.4, 4.2; 0, 3.6; 0, 10; 0, 1; -0.5, 3.5];
    
    for i_dataset = 1:numel(dataset_fields)
        curr_field = dataset_fields{i_dataset};
        curr_dataset = prec_by_dataset.(curr_field).dataset;
        
        fprintf('   Plotting %s... ', curr_dataset);
        
        gen = prec_by_dataset.(curr_field).params.gen;
        est = prec_by_dataset.(curr_field).params.est;
        
        % Apply censoring
        est(est(:, end) < 2, 3) = NaN;  % No gamma if kappa < 2
        est(est(:, end) == 0, 2) = NaN;  % No lambda if kappa < 1
        
        dataset_color = dataset_colors.(strrep(curr_dataset, '-', '_'));
        
        % Plot parameters
        ii = 1;
        for i_param = [4, 1:3, 5]  % Order: kappa, alpha, beta, gamma, lambda
            subplot_idx = (i_dataset - 1) * 5 + ii;
            subplot(n_datasets, 5, subplot_idx);
            hold on;
            
            % Identity line
            plot(param_lims(i_param, :), param_lims(i_param, :), 'k--', 'Color', [0.7 0.7 0.7]);
            
            % Scatter
            if i_param == 5
                noise = normrnd(0, 0.15, size(gen, 1), 2);
                scatter(gen(:, i_param) + noise(:, 1), est(:, i_param) + noise(:, 2), ...
                        'filled', 'MarkerFaceAlpha', 0.3, 'MarkerEdgeAlpha', 0.5, ...
                        'MarkerEdgeColor', dataset_color, 'CData', dataset_color);
            else
                scatter(gen(:, i_param), est(:, i_param), ...
                        'filled', 'MarkerFaceAlpha', 0.3, 'MarkerEdgeAlpha', 0.5, ...
                        'MarkerEdgeColor', dataset_color, 'CData', dataset_color);
            end
            
            % Correlation
            [r, p] = corr(gen(:, i_param), est(:, i_param), 'rows', 'pairwise');
            
            % Format
            title(sprintf('%s (r=%.2f)', params{i_param}, r), 'FontSize', 10);
            xlim(param_lims(i_param, :));
            ylim(param_lims(i_param, :));
            axis square;
            
            if ii == 1
                ylabel(strrep(curr_dataset, '-', ' '), 'FontSize', 11, 'FontWeight', 'bold');
            end
            
            if i_dataset == n_datasets
                xlabel('Generating', 'FontSize', 10);
            end
            
            if i_dataset == 1 && ii == 1
                ylabel({'Recovered', strrep(curr_dataset, '-', ' ')}, 'FontSize', 11, 'FontWeight', 'bold');
            end
            
            box off;
            ii = ii + 1;
        end
        
        fprintf('done\n');
    end
    
    sgtitle('Parameter Recovery by Dataset', 'FontSize', 15, 'FontWeight', 'bold');
    
    exportgraphics(fig, fullfile(results_folder, 'parameter_recovery_by_dataset.png'), 'Resolution', 300);
    fprintf('   ✓ Parameter recovery plotting complete\n\n');
    close(fig);
    
catch e
    fprintf('\n   ❌ Error in parameter recovery plotting: %s\n', e.message);
    fprintf('   (This is expected if fitting script hasn''t been updated yet)\n\n');
end

fprintf('========================================\n');
fprintf('DATASET-SPECIFIC ANALYSES COMPLETE\n');
fprintf('========================================\n\n');

%% Helper function for behavioral signatures
function [meanBR_real, SEBR_real, meanBR_sim, SEBR_sim] = compute_behavioral_signatures(data_real, data_sim)
    % Compute best-response signatures
    % Returns: [3 levels × 3 bot_levels × 40 trials]
    
    winSize = 5;
    
    for data_type = 1:2
        if data_type == 1
            data = data_real;
        else
            data = data_sim;
        end
        
        % Initialize
        BR = nan(40, 3, 3);  % trials × bot_levels × BR_types
        
        for bot_level = 0:2
            data_bot = data(data.bot_level == bot_level, :);
            n_trials = height(data_bot) - 1;
            
            if n_trials < 1, continue; end
            
            % BR to level 0 (opponent's last action + 1)
            predBR1 = mod(data_bot.choice_other(1:end-1), 3) + 1;
            
            % BR to level 1 (own last action - 1)
            predBR2 = data_bot.choice_own(1:end-1) - 1;
            predBR2(predBR2 == 0) = 3;
            
            % BR to level 2 (opponent's last action)
            predBR3 = data_bot.choice_other(1:end-1);
            
            % Check correctness
            BR(1:n_trials, bot_level + 1, 1) = (predBR1 == data_bot.choice_own(2:end));
            BR(1:n_trials, bot_level + 1, 2) = (predBR2 == data_bot.choice_own(2:end));
            BR(1:n_trials, bot_level + 1, 3) = (predBR3 == data_bot.choice_own(2:end));
        end
        
        % Moving average
        meanBR = nan(3, 3, 40);
        SEBR = nan(3, 3, 40);
        
        for trial = 1:40
            if trial < winSize
                range = 1:trial;
            else
                range = trial - winSize + 1:trial;
            end
            
            for bot_level = 1:3
                for br_type = 1:3
                    vals = BR(range, bot_level, br_type);
                    vals = vals(~isnan(vals));
                    
                    if ~isempty(vals)
                        meanBR(br_type, bot_level, trial) = mean(vals) - 0.33;  % Relative to chance
                        SEBR(br_type, bot_level, trial) = 1.96 * std(vals) / sqrt(length(vals));
                    end
                end
            end
        end
        
        if data_type == 1
            meanBR_real = meanBR;
            SEBR_real = SEBR;
        else
            meanBR_sim = meanBR;
            SEBR_sim = SEBR;
        end
    end
end


%% Block Consistency Analysis (One-block vs Three-block fitting)

fprintf('  → Block consistency analysis\n');

try
    % ========================================================================
    % DEBUG: Check input structure
    % ========================================================================
    fprintf('\n    [DEBUG] Checking input structures...\n');
    fprintf('      fits type: %s\n', class(fits));
    fprintf('      fits length: %d\n', numel(fits));
    fprintf('      fits(1).subj type: %s\n', class(fits(1).subj));
    fprintf('      fits(1).subj length: %d\n', numel(fits(1).subj));
    fprintf('      fits(1).subj fields: %s\n', strjoin(fieldnames(fits(1).subj), ', '));
    if numel(fits(1).subj) > 0
        fprintf('      fits(1).subj(1) data fields: %s\n', strjoin(fieldnames(fits(1).subj(1).data), ', '));
        
        % Check if dataset is cell or char
        dataset_example = fits(1).subj(1).data.dataset;
        fprintf('      fits(1).subj(1).data.dataset type: %s\n', class(dataset_example));
        if iscell(dataset_example)
            fprintf('      fits(1).subj(1).data.dataset: %s (cell)\n', dataset_example{1});
        else
            fprintf('      fits(1).subj(1).data.dataset: %s (char)\n', dataset_example);
        end
    end
    
    % Load the pooled fits (current approach)
    fits_pooled = fits(1);
    fprintf('      fits_pooled.subj length: %d\n', numel(fits_pooled.subj));
    
    % ========================================================================
    % Refit with separate blocks
    % ========================================================================
    fprintf('\n    [DEBUG] Loading data for separate block fitting...\n');
    load(fullfile(project_folder,'data','llm_data_5.mat'));
    fprintf('      Data loaded, height: %d\n', height(data));
    
    % Restructure: keep blocks separate
    if ismember('n_trials', data.Properties.VariableNames)
        data.n_trials = [];
    end
    data.n_blocks(:) = 3;  % Signal to fit separate parameters per block
    fprintf('      Set n_blocks = 3\n');
    
    % Convert WITH block_var to fit separate parameters per block
    fprintf('      Converting to struct with block_var...\n');
    data_sep = mn_table2struct(data, 'subjID', 'remove_redundancy', ...
                               'exceptions', {'choice_own','choice_other','missing'}, ...
                               'block_var', 'block');
    fprintf('      Converted, length: %d subjects\n', numel(data_sep));
    
    fprintf('    Fitting model with separate parameters per block...\n');
    model = BAKR_2024_CHASE_config('CH','fitted',3,'RW-freq');
    fits_separate = mn_fit(data_sep, model);
    
    fprintf('\n    [DEBUG] Separate fitting complete\n');
    fprintf('      fits_separate.subj length: %d\n', numel(fits_separate.subj));
    fprintf('      fits_separate.subj(1) fields: %s\n', strjoin(fieldnames(fits_separate.subj(1)), ', '));
    fprintf('      fits_separate.subj(1).params type: %s\n', class(fits_separate.subj(1).params));
    if isstruct(fits_separate.subj(1).params)
        fprintf('      fits_separate.subj(1).params length: %d\n', numel(fits_separate.subj(1).params));
    end
    
    % ========================================================================
    % Extract parameters for comparison
    % ========================================================================
    param_names = {'alpha', 'beta', 'gamma', 'lambda', 'kappa'};
    datasets = {'DEEPSEEK-NORMAL', 'DEEPSEEK-SCOT', 'GPT-NORMAL', 'HUMAN'};
    
    fprintf('\n    [DEBUG] Starting figure generation...\n');
    
    % Create figure
    figure('Position', [100, 100, 1400, 1000]);
    
    for i_dataset = 1:numel(datasets)
        fprintf('      Processing dataset %d/%d: %s\n', i_dataset, numel(datasets), datasets{i_dataset});
        
        % ========================================================================
        % DEBUG: Check indexing - SAFE extraction
        % ========================================================================
        fprintf('        [DEBUG] Getting dataset names...\n');
        
        % Extract dataset names properly (handle cell arrays)
        dataset_names_pooled = cell(numel(fits_pooled.subj), 1);
        for i = 1:numel(fits_pooled.subj)
            ds = fits_pooled.subj(i).data.dataset;
            if iscell(ds)
                dataset_names_pooled{i} = ds{1};
            else
                dataset_names_pooled{i} = ds;
            end
        end
        fprintf('          dataset_names_pooled extracted: %d names\n', numel(dataset_names_pooled));
        
        dataset_names_separate = cell(numel(fits_separate.subj), 1);
        for i = 1:numel(fits_separate.subj)
            ds = fits_separate.subj(i).data.dataset;
            if iscell(ds)
                dataset_names_separate{i} = ds{1};
            else
                dataset_names_separate{i} = ds;
            end
        end
        fprintf('          dataset_names_separate extracted: %d names\n', numel(dataset_names_separate));
        
        % Find subjects in this dataset
        idx_pooled = strcmp(dataset_names_pooled, datasets{i_dataset});
        fprintf('          idx_pooled: %d subjects found\n', sum(idx_pooled));
        
        idx_separate = strcmp(dataset_names_separate, datasets{i_dataset});
        fprintf('          idx_separate: %d subjects found\n', sum(idx_separate));
        
        n_subj = sum(idx_pooled);
        
        for i_param = 1:5
            fprintf('          Parameter %d/%d: %s\n', i_param, 5, param_names{i_param});
            
            subplot(4, 5, (i_dataset-1)*5 + i_param);
            hold on;
            
            % ========================================================================
            % DEBUG: Extract pooled estimates
            % ========================================================================
            fprintf('            [DEBUG] Extracting pooled estimates...\n');
            
            % SAFE extraction
            params_pooled = NaN(n_subj, 1);
            subj_pooled_indices = find(idx_pooled);
            for i_s = 1:n_subj
                params_pooled(i_s) = fits_pooled.subj(subj_pooled_indices(i_s)).params.(param_names{i_param});
            end
            fprintf('              params_pooled extracted: %d values\n', numel(params_pooled));
            
            % ========================================================================
            % DEBUG: Extract separate estimates
            % ========================================================================
            fprintf('            [DEBUG] Extracting separate estimates...\n');
            
            params_sep_mean = NaN(n_subj, 1);
            subj_separate_indices = find(idx_separate);
            
            for i_subj = 1:n_subj
                if i_subj <= 2  % Only debug first 2 subjects
                    fprintf('              Subject %d/%d\n', i_subj, n_subj);
                end
                
                curr_params = fits_separate.subj(subj_separate_indices(i_subj)).params;
                
                if i_subj == 1
                    fprintf('                curr_params type: %s, length: %d\n', class(curr_params), numel(curr_params));
                end
                
                block_params = NaN(1, 3);
                for i_block = 1:3
                    block_params(i_block) = curr_params(i_block).(param_names{i_param});
                end
                params_sep_mean(i_subj) = mean(block_params);
                
                if i_subj == 1
                    fprintf('                block_params: [%.3f, %.3f, %.3f]\n', block_params(1), block_params(2), block_params(3));
                    fprintf('                mean: %.3f\n', params_sep_mean(i_subj));
                end
            end
            fprintf('              params_sep_mean extracted: %d values\n', numel(params_sep_mean));
            
            % ========================================================================
            % Plotting
            % ========================================================================
            
            % Plot identity line
            lims = [min([params_pooled; params_sep_mean]), max([params_pooled; params_sep_mean])];
            plot(lims, lims, 'k--', 'LineWidth', 1, 'Color', [0.7 0.7 0.7]);
            
            % Scatter plot
            dataset_colors = struct('DEEPSEEK_NORMAL', [0.3, 0.7, 0.5], ...
                                   'DEEPSEEK_SCOT', [0.2, 0.6, 0.4], ...
                                   'GPT_NORMAL', [0.8, 0.5, 0.3], ...
                                   'HUMAN', [0.6, 0.3, 0.5]);
            color = dataset_colors.(strrep(datasets{i_dataset}, '-', '_'));
            
            scatter(params_pooled, params_sep_mean, 40, color, 'filled', ...
                   'MarkerFaceAlpha', 0.6, 'MarkerEdgeAlpha', 0.8);
            
            % Compute correlation
            [r, p] = corr(params_pooled, params_sep_mean, 'rows', 'pairwise');
            
            % Format
            xlabel('Pooled (120 trials)', 'FontSize', 9);
            ylabel('Separate (mean of 3×40)', 'FontSize', 9);
            title(sprintf('%s (r=%.2f)', param_names{i_param}, r), 'FontSize', 10);
            axis square;
            xlim(lims); ylim(lims);
            
            % Add dataset label on leftmost panel
            if i_param == 1
                text(-0.5, 0.5, datasets{i_dataset}, 'Units', 'normalized', ...
                     'Rotation', 90, 'HorizontalAlignment', 'center', ...
                     'FontSize', 11, 'FontWeight', 'bold');
            end
        end
    end
    
    sgtitle('Block Consistency: Pooled vs. Separate Block Fitting', ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    % Save figure
    exportgraphics(gcf, fullfile(output_dir, 'block_consistency_analysis.png'), 'Resolution', 300);
    fprintf('    ✓ Saved: block_consistency_analysis.png\n');
    
    % ========================================================================
    % STATISTICAL TESTS
    % ========================================================================
    
    fprintf('\n    [DEBUG] Starting statistical tests...\n');
    
    % Print correlation summary
    fprintf('\n    Block consistency correlations (pooled vs. mean of separate):\n');
    for i_dataset = 1:numel(datasets)
        fprintf('      %s:\n', datasets{i_dataset});
        
        % Safe indexing - extract dataset names properly
        dataset_names_pooled = cell(numel(fits_pooled.subj), 1);
        for i = 1:numel(fits_pooled.subj)
            ds = fits_pooled.subj(i).data.dataset;
            if iscell(ds)
                dataset_names_pooled{i} = ds{1};
            else
                dataset_names_pooled{i} = ds;
            end
        end
        
        dataset_names_separate = cell(numel(fits_separate.subj), 1);
        for i = 1:numel(fits_separate.subj)
            ds = fits_separate.subj(i).data.dataset;
            if iscell(ds)
                dataset_names_separate{i} = ds{1};
            else
                dataset_names_separate{i} = ds;
            end
        end
        
        idx_pooled = strcmp(dataset_names_pooled, datasets{i_dataset});
        idx_separate = strcmp(dataset_names_separate, datasets{i_dataset});
        
        n_subj = sum(idx_pooled);
        subj_pooled_indices = find(idx_pooled);
        subj_separate_indices = find(idx_separate);
        
        for i_param = 1:5
            % Extract pooled
            params_pooled = NaN(n_subj, 1);
            for i_s = 1:n_subj
                params_pooled(i_s) = fits_pooled.subj(subj_pooled_indices(i_s)).params.(param_names{i_param});
            end
            
            % Extract separate
            params_sep_mean = NaN(n_subj, 1);
            for i_s = 1:n_subj
                block_params = NaN(1, 3);
                for i_block = 1:3
                    block_params(i_block) = fits_separate.subj(subj_separate_indices(i_s)).params(i_block).(param_names{i_param});
                end
                params_sep_mean(i_s) = mean(block_params);
            end
            
            [r, p] = corr(params_pooled, params_sep_mean, 'rows', 'pairwise');
            fprintf('        %s: r=%.3f (p=%.4f)\n', param_names{i_param}, r, p);
        end
    end
    
    % Test parameter consistency across blocks (within-subject)
    fprintf('\n    Testing parameter consistency across blocks (ANOVA):\n');
    for i_dataset = 1:numel(datasets)
        fprintf('      %s:\n', datasets{i_dataset});
        
        % Extract dataset names properly
        dataset_names_separate = cell(numel(fits_separate.subj), 1);
        for i = 1:numel(fits_separate.subj)
            ds = fits_separate.subj(i).data.dataset;
            if iscell(ds)
                dataset_names_separate{i} = ds{1};
            else
                dataset_names_separate{i} = ds;
            end
        end
        
        idx_subj = strcmp(dataset_names_separate, datasets{i_dataset});
        n_subj = sum(idx_subj);
        subj_indices = find(idx_subj);
        
        for i_param = 1:5
            % Extract parameters for each block (rows=subjects, cols=blocks)
            params_block = NaN(n_subj, 3);
            for i_s = 1:n_subj
                for i_block = 1:3
                    params_block(i_s, i_block) = ...
                        fits_separate.subj(subj_indices(i_s)).params(i_block).(param_names{i_param});
                end
            end
            
            % One-way ANOVA: Are parameters different across blocks?
            [p, tbl, stats] = anova1(params_block, [], 'off');
            F_stat = tbl{2,5};  % F-statistic
            
            % Interpretation
            if p < 0.05
                sig_marker = '***';
            elseif p < 0.10
                sig_marker = '*';
            else
                sig_marker = '';
            end
            
            fprintf('        %s: F(2,%d)=%.2f, p=%.3f %s\n', ...
                    param_names{i_param}, tbl{3,3}, F_stat, p, sig_marker);
        end
    end
    
    % Compare model fit: pooled vs. separate
    fprintf('\n    Model fit comparison:\n');
    
    % Overall (all subjects)
    aic_pooled_vec = NaN(numel(fits_pooled.subj), 1);
    for i = 1:numel(fits_pooled.subj)
        aic_pooled_vec(i) = fits_pooled.subj(i).optim.AIC;
    end
    aic_pooled = mean(aic_pooled_vec);
    
    aic_separate_vec = NaN(numel(fits_separate.subj), 1);
    for i = 1:numel(fits_separate.subj)
        aic_separate_vec(i) = fits_separate.subj(i).optim.AIC;
    end
    aic_separate = mean(aic_separate_vec);
    
    fprintf('      ALL: Pooled AIC=%.2f, Separate AIC=%.2f (Δ=%.2f)\n', ...
            aic_pooled, aic_separate, aic_separate - aic_pooled);
    
    % By dataset
    for i_dataset = 1:numel(datasets)
        % Extract dataset names properly
        dataset_names_pooled = cell(numel(fits_pooled.subj), 1);
        for i = 1:numel(fits_pooled.subj)
            ds = fits_pooled.subj(i).data.dataset;
            if iscell(ds)
                dataset_names_pooled{i} = ds{1};
            else
                dataset_names_pooled{i} = ds;
            end
        end
        
        dataset_names_separate = cell(numel(fits_separate.subj), 1);
        for i = 1:numel(fits_separate.subj)
            ds = fits_separate.subj(i).data.dataset;
            if iscell(ds)
                dataset_names_separate{i} = ds{1};
            else
                dataset_names_separate{i} = ds;
            end
        end
        
        idx_pooled = strcmp(dataset_names_pooled, datasets{i_dataset});
        idx_separate = strcmp(dataset_names_separate, datasets{i_dataset});
        
        % Extract AICs safely
        subj_pooled_indices = find(idx_pooled);
        aic_pooled_ds_vec = NaN(numel(subj_pooled_indices), 1);
        for i = 1:numel(subj_pooled_indices)
            aic_pooled_ds_vec(i) = fits_pooled.subj(subj_pooled_indices(i)).optim.AIC;
        end
        aic_pooled_ds = mean(aic_pooled_ds_vec);
        
        subj_separate_indices = find(idx_separate);
        aic_separate_ds_vec = NaN(numel(subj_separate_indices), 1);
        for i = 1:numel(subj_separate_indices)
            aic_separate_ds_vec(i) = fits_separate.subj(subj_separate_indices(i)).optim.AIC;
        end
        aic_separate_ds = mean(aic_separate_ds_vec);
        
        fprintf('      %s: Pooled AIC=%.2f, Separate AIC=%.2f (Δ=%.2f)\n', ...
                datasets{i_dataset}, aic_pooled_ds, aic_separate_ds, ...
                aic_separate_ds - aic_pooled_ds);
    end
    
    fprintf('\n  ✓ Block consistency analysis complete\n\n');
    
catch e
    fprintf('\n    ❌ Error occurred: %s\n', e.message);
    fprintf('    Error identifier: %s\n', e.identifier);
    fprintf('    Stack trace:\n');
    for i = 1:length(e.stack)
        fprintf('      [%d] %s (line %d)\n', i, e.stack(i).name, e.stack(i).line);
    end
    fprintf('    Continuing with remaining analyses...\n\n');
end