function counts = BAKR_2024_model_recovery_plot(sim_fits,idx_sort)

if nargin < 2
    idx_sort = 1:numel(sim_fits);
end

% do model comparison for each generated dataset
for i_model_gen = 1:numel(sim_fits)

    curr_subj = (floor([sim_fits(1).subj.subjID]/100) == i_model_gen);

    clear curr_fits
    for i_model_fit = 1:numel(sim_fits)
        curr_fits(i_model_fit) = sim_fits(i_model_fit);
        curr_fits(i_model_fit).subj(~curr_subj) = [];
    end

    [~,out(i_model_gen)] = mn_compare(curr_fits,'flag_plot',0);
    count_mat(i_model_gen,:) = out(i_model_gen).rand.AIC.counts;

end
count_mat = count_mat(idx_sort,idx_sort);
counts = count_mat./sum(count_mat,2);

% plot
imagesc(counts,[0,1]);

ticklabels = arrayfun(@(model) model.model.name,sim_fits(idx_sort),'UniformOutput',false);
xticks(1:numel(sim_fits)); xticklabels(ticklabels); xtickangle(45);
yticks(1:numel(sim_fits)); yticklabels(ticklabels);

ylabel('Generating model');
xlabel('Best-fitting model');

cb = colorbar;
yl = ylabel(cb,'Model count (%)','FontSize',12,'Rotation',270);
axis square
colormap bone
b = colormap;
colormap(flipud(b));

end