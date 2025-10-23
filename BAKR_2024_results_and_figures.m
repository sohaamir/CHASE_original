%
% Results and figures for Buergi, Aydogan, Konovalov, & Ruff (2024):
% "A neural fingerprint of adaptive mentalization" 
%
% To re-create the files used herein, use run_model_fitting and run_fmri_analysis
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

    exportgraphics(gca,'model_comparison.png','Resolution',300);

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

%% level distribution in human-human gameplay

load(fullfile(project_folder,'results','supplementary','fits_LK_perblock.mat'),'fits_LK_perblock');
ks = arrayfun(@(subj) [subj.params.kappa],fits_LK_perblock.subj,'UniformOutput',0);

datasets = arrayfun(@(subj) subj.data.dataset,fits_LK_perblock.subj);
idx_RPS3 = contains(datasets,'Human, RPS-3');
idx_RPS4 = contains(datasets,'Human, RPS-4');

figure;
subplot(1,3,1);
histogram([ks{idx_RPS3 | idx_RPS4}],'Normalization','probability');ylim([0,0.4]); title('All human data'); ylabel('Frequency'); xlabel('Levels');
subplot(1,3,2);
histogram([ks{idx_RPS4}],'Normalization','probability'); ylim([0,0.4]); title('Only RPS-4'); xlabel('Levels');
subplot(1,3,3);
histogram([ks{idx_RPS3}],'Normalization','probability'); ylim([0,0.4]); title('Only RPS-3'); xlabel('Levels');

% exportgraphics(gcf,'level_distribution_human_human.png');

%% LR recovery

figure; 
load(fullfile(results_folder,'supplementary','model_recovery_LR.mat'),'sim_fits');  % ✓ FIX
BAKR_2024_model_recovery_plot(sim_fits,[1,2,4,3]);

% exportgraphics(gcf,'model_recovery_LR.png','Resolution',300);

%% LR comparisons

load(fullfile(project_folder,'results','supplementary','model_comparison_LR.mat'),'fits_LR');
fits_LR(3:4) = [];

% per opponent type
figure;
subplot(1,2,1); 
[~,stats_opp_type] = mn_compare(fits_LR,'group','opp_type','use_current_fig');
title('Opponent type'); xlabel('PXP');

% adjust colors
a = gca;
a.Children = flipud(a.Children);
f = gcf;
f.Children(1).String = fliplr(f.Children(1).String);
legend('Bot','Human');
col =  0.5*colors(2,:) + 0.5*(colors(3,:));
f.Children(2).Children(2).FaceColor = col;
f.Children(2).Children(2).EdgeColor = col;
f.Children(2).Children(2).EdgeAlpha = .8;
f.Children(2).Children(2).FaceAlpha = .7;
f.Children(2).Children(1).EdgeAlpha = .7;
f.Children(2).Children(1).FaceAlpha = .6;

% per dataset
subplot(1,2,2);
[~,stats_datasets] = mn_compare(fits_LR,'group','dataset','use_current_fig');
title('Individual datasets'); xlabel('PXP');
f.Children(1).FontSize = 8;

% stats
pxp_opp_type = arrayfun(@(dataset) dataset.rand.AIC.pxp(1),stats_opp_type)
pxp_dataset = arrayfun(@(dataset) dataset.rand.AIC.pxp(1),stats_datasets)

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
