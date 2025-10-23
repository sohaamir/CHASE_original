function BAKR_2024_posterior_predictive_check(results_folder,type,panels)
% Changed parameter name from project_folder to results_folder

load(fullfile(results_folder,'simulations.mat'));

% Loop over models to plot: %'CHASE_CH-leaky_static_k-fitted_max-3_RW-freq';%'CHASE_LK_static_k-fixed_max-0_EWA-single';%'CHASE_LK_static_k-fixed_max-1_RW-freq';%'CHASE_LK_static_k-fixed_max-0_RW-reward';%'CHASE_LK_static_k-fixed_max-0_EWA-full'; %'CHASE_CH-leaky_static_k-fitted_max-3_RW-freq';

models = unique({sims.model});

switch type
    case 'main', models_to_plot = models(1);
    case 'all', models_to_plot = models;
end

for uniqueModels = models_to_plot

    modelInt = uniqueModels{1}; %'CHASE_CH-leaky_static_k-fitted_max-3_RW-freq';

    % rename models - UPDATED by AS to handle both old and new naming conventions
    if contains(modelInt,'CHASE_CH-leaky_static') || strcmp(modelInt,'CHASE')
        modelName = 'CHASE';
    elseif contains(modelInt,'max-1_RW-freq') || strcmp(modelInt,'Ficticious play')
        modelName = 'Ficticious play';
    elseif contains(modelInt,'max-0_RW-reward') || strcmp(modelInt,'Reward learner')
        modelName = 'Reward learner';
    elseif contains(modelInt,'max-0_EWA-single') || strcmp(modelInt,'Self-tuning EWA')
        modelName = 'Self-tuning EWA';
    elseif contains(modelInt,'max-0_EWA-full') || strcmp(modelInt,'Full EWA')
        modelName = 'Full EWA';
    elseif contains(modelInt,'ToMk') || strcmp(modelInt,'ToMk')
        modelName = 'ToMk';
    else
        % Fallback: use the model name as-is if no match found
        modelName = modelInt;
        warning('Unknown model name format: %s. Using as-is.', modelInt);
    end

    set(groot,'DefaultAxesLineWidth',2)
    set(groot,'defaultAxesFontSize',12)

    % Fsize = 24;
    Fsize = 10;

    colorbase(1,:) = [50,130,150]./255;
    colorbase(2,:) = [60,130,100]./255;
    colorbase(3,:) = [130,170,80]./255; % [180,200,60]/255
    colorbase(4:6,:) = colorbase(1:3,:);
    colorbase(7,:) = [90 90 90]./255;

    % alternative color base
    % colorbase(1,1:3) = [243 83 127]./255; % R243 G83 B127
    % colorbase(2,1:3) = [0 30 124]./255;     % R0 G30 B124
    % colorbase(3,1:3) = [124 160 35]./255;     % R124 G160 B35
    % colorbase(4:6,:) = colorbase(1:3,:); % R243 G83 B127
    % colorbase(7,1:3) = [102 102 102]./255;  % R102 G102 B102

    winSize = 5;

    new_G = new_T;


    %%

    numTrials = sims(1).n_trials;
    tempT = [];
    new_Sim = [];
    for ii = 1:numel(sims)

        tempT = array2table([repmat(sims(ii).subjID,numTrials,1), sims(ii).trial, sims(ii).choice_own,  sims(ii).choice_other, sims(ii).score_own, sims(ii).score_other, repelem(sims(ii).bot_level, 40)  ],'VariableNames',["subjID","trial","choice_own","choice_other","score_own","score_other","bot_level"] );

        tempT = [tempT, cell2table(repmat({sims(ii).model},numTrials,1),'VariableNames',"model") ];
        new_Sim = [new_Sim; tempT];

    end


    new_Sim(~contains(new_Sim.model,modelInt),:) = [];



    %%
    [predBR1 predBR2 predBR3] = deal(-10.*ones(numel(new_T.subjID),3));
    [SimpredBR1 SimpredBR2 SimpredBR3] = deal(-10.*ones(numel(new_Sim.subjID),3));


    for numLevel = 1:3
        dataSub = new_T( new_T.bot_level== numLevel -1 ,:);

        numTrials = numel(dataSub.subjID(:,1))-1;

        % strategy BR to level 0
        predBR1(1:numTrials,numLevel) = (dataSub.choice_other(1:end-1,:)) + 1;
        predBR1(predBR1(1:numTrials,numLevel)==4,numLevel) = 1;

        % strategy BR to level 1
        predBR2(1:numTrials,numLevel) = (dataSub.choice_own(1:end-1,:)) - 1;
        predBR2(predBR2(1:numTrials,numLevel)==0,numLevel) = 3;

        % strategy BR to level 2
        predBR3(1:numTrials,numLevel) = (dataSub.choice_other(1:end-1,:));

        BR1(1:numTrials,numLevel) = predBR1(1:numTrials,numLevel) == dataSub.choice_own(2:end,:);
        BR2(1:numTrials,numLevel) = predBR2(1:numTrials,numLevel) == dataSub.choice_own(2:end,:);
        BR3(1:numTrials,numLevel) = predBR3(1:numTrials,numLevel) == dataSub.choice_own(2:end,:);


        %Simulated Data
        dataSim = new_Sim( new_Sim.bot_level== numLevel -1 ,:);
        SimNumTrials = numel(dataSim.subjID(:,1))-1;
        % strategy BR to level 0
        SimpredBR1(1:SimNumTrials,numLevel) = (dataSim.choice_other(1:end-1,:)) + 1;
        SimpredBR1(SimpredBR1(1:SimNumTrials,numLevel)==4,numLevel) = 1;
        % strategy BR to level 1
        SimpredBR2(1:SimNumTrials,numLevel) = (dataSim.choice_own(1:end-1,:)) - 1;
        SimpredBR2(SimpredBR2(1:SimNumTrials,numLevel)==0,numLevel) = 3;
        % strategy BR to level 2
        SimpredBR3(1:SimNumTrials,numLevel) = (dataSim.choice_other(1:end-1,:));

        SimBR1(1:SimNumTrials,numLevel) = SimpredBR1(1:SimNumTrials,numLevel) == dataSim.choice_own(2:end,:);
        SimBR2(1:SimNumTrials,numLevel) = SimpredBR2(1:SimNumTrials,numLevel) == dataSim.choice_own(2:end,:);
        SimBR3(1:SimNumTrials,numLevel) = SimpredBR3(1:SimNumTrials,numLevel) == dataSim.choice_own(2:end,:);



        for ii = 1:40

            if ii < winSize
                % Start with smaller ranges, e.g., 1:2, 1:3, ..., 1:winSize
                range = 1:ii;
            else
                % Slide the window forward, e.g., 2:6, 3:7, etc.
                range = ii-winSize+1:ii;
            end


            tempTrials = ismember(dataSub.trial(:,1) , range);
            tempTrials = tempTrials(2:end,1);


            meanBR1(ii,numLevel) = mean(BR1(tempTrials(1:end,1),numLevel));
            SEBR1(ii,numLevel) = 1.96*std(BR1(tempTrials(1:end,1),numLevel)) ./ sqrt(sum(tempTrials(1:end,1)));

            meanBR2(ii,numLevel) = mean(BR2(tempTrials(1:end,1),numLevel));
            SEBR2(ii,numLevel) = 1.96*std(BR2(tempTrials(1:end,1),numLevel)) ./ sqrt(sum(tempTrials(1:end,1)));


            meanBR3(ii,numLevel) = mean(BR3(tempTrials(1:end,1),numLevel));
            SEBR3(ii,numLevel) = 1.96*std(BR3(tempTrials(1:end,1),numLevel)) ./ sqrt(sum(tempTrials(1:end,1)));


            SIMtempTrials = ismember(dataSim.trial(:,1) , range);
            SIMtempTrials = SIMtempTrials(2:end);


            SIMmeanBR1(ii,numLevel) = mean(SimBR1(SIMtempTrials(1:end,1),numLevel));
            SIMSEBR1(ii,numLevel) = 1.96*std(SimBR1(SIMtempTrials(1:end,1),numLevel)) ./ sqrt(sum(SIMtempTrials(1:end,1)));

            SIMmeanBR2(ii,numLevel) = mean(SimBR2(SIMtempTrials(1:end,1),numLevel));
            SIMSEBR2(ii,numLevel) = 1.96*std(SimBR2(SIMtempTrials(1:end,1),numLevel)) ./ sqrt(sum(SIMtempTrials(1:end,1)));


            SIMmeanBR3(ii,numLevel) = mean(SimBR3(SIMtempTrials(1:end,1),numLevel));
            SIMSEBR3(ii,numLevel) = 1.96*std(SimBR3(SIMtempTrials(1:end,1),numLevel)) ./ sqrt(sum(SIMtempTrials(1:end,1)));



        end

        %

    end

    %% The plots
    % Define the trial length t (1 to 40)
    t = 1:40;

    SIMmeanBR1 = [ SIMmeanBR1]';
    SIMSEBR1   = [ SIMSEBR1]';
    SIMmeanBR2 = [ SIMmeanBR2]';
    SIMSEBR2   = [ SIMSEBR2]';
    SIMmeanBR3 = [ SIMmeanBR3]';
    SIMSEBR3   = [ SIMSEBR3]';

    meanBR1    = [ meanBR1]';
    SEBR1      = [ SEBR1]';
    meanBR2    = [ meanBR2]';
    SEBR2      = [ SEBR2]';
    meanBR3    = [ meanBR3]';
    SEBR3      = [ SEBR3]';




    if strcmp(type,'all')
        fig = figure('Name','Behavioral and model-based evidence for adaptive mentalization.',...
            'Units','normalized','Position',[0,0,1,1]);
    end

    for numLevel = 1:3

        SimMeanTrue = [SIMmeanBR1(1,:); SIMmeanBR2(2,:); SIMmeanBR3(3,:) ]-.33;
        SIMSEBRTrue = [SIMSEBR1(1,:); SIMSEBR2(2,:); SIMSEBR3(3,:) ];

        SimMeanFalse = 1-[SIMmeanBR1(1,:); SIMmeanBR2(2,:); SIMmeanBR3(3,:) ] - .66;
        SIMSEBRFalse = [mean(SIMSEBR1(2:3,:)); mean(SIMSEBR2(2:3,:)); mean(SIMSEBR3(2:3,:)) ];

        MeanTrue = [meanBR1(1,:); meanBR2(2,:); meanBR3(3,:) ] - .33;
        SEBRTrue = [SEBR1(1,:); SEBR2(2,:); SEBR3(3,:) ];

        MeanFalse = 1-[meanBR1(1,:); meanBR2(2,:); meanBR3(3,:) ] - .66;
        SEBRFalse = [mean(SEBR1(2:3,:)); mean(SEBR2(2:3,:)); mean(SEBR3(2:3,:)) ];


        % Create the figure
        set(gcf, 'Color', 'w')

        switch type
            case 'main', subplot(panels{1},panels{2},panels{3}(2*(numLevel-1)+1:2*(numLevel-1)+2));
            case 'all', subplot(1,3,numLevel);
        end

        % Plot the shaded area for meanPlayModel +/- SEPlayModel
        fill([t fliplr(t)], [SimMeanTrue(numLevel,:) + SIMSEBRTrue(numLevel,:), fliplr(SimMeanTrue(numLevel,:) - SIMSEBRTrue(numLevel,:))], ...
            colorbase(numLevel,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');  % Blue shaded area

        hold on;

        % Plot the mean line for PlayModel
        plot(t, SimMeanTrue(numLevel,:),'Color', colorbase(numLevel,:), 'LineWidth', 2);  % Blue line

        % Plot the shaded area for meanNotPlayModel +/- SENotPlayModel
        fill([t fliplr(t)], [SimMeanFalse(numLevel,:) + SIMSEBRFalse(numLevel,:), fliplr(SimMeanFalse(numLevel,:) - SIMSEBRFalse(numLevel,:))], ...
            colorbase(7,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');  % Red shaded area

        % Plot the mean line for NotPlayModel
        plot(t, SimMeanFalse(numLevel,:), 'Color', colorbase(7,:), 'LineWidth', 2);  % Red line


        % d = errorbar(t,MeanTrue(numLevel,:),SEBRTrue(numLevel,:),"LineStyle","none","CapSize",5,"Marker","diamond","MarkerSize",8,"Color",colorbase(numLevel,:),'MarkerEdgeColor',colorbase(numLevel,:));
        % errorbar(t,MeanFalse(numLevel,:),SEBRFalse(numLevel,:),"LineStyle","none","CapSize",5,"Marker","diamond","MarkerSize",8,"Color",colorbase(7,:),'MarkerEdgeColor',colorbase(7,:))
        d = errorbar(t,MeanTrue(numLevel,:),SEBRTrue(numLevel,:),"LineWidth",0.75,"LineStyle","none","CapSize",2,"Marker","o","MarkerSize",2,"Color",colorbase(numLevel,:),'MarkerEdgeColor',colorbase(numLevel,:),'MarkerFaceColor',colorbase(numLevel,:));
        errorbar(t,MeanFalse(numLevel,:),SEBRFalse(numLevel,:),"LineWidth",0.75,"LineStyle","none","CapSize",2,"Marker","o","MarkerSize",2,"Color",colorbase(7,:),'MarkerEdgeColor',colorbase(7,:),'MarkerFaceColor',colorbase(numLevel,:))

        % Labels and title
        xlabel('Trial','FontSize',Fsize*1.2);
        if numLevel <2
            ylabel('Frequency relative to chance','FontSize',Fsize*1.2);
        end
        title(['Bot \itk\rm\bf = ' num2str(numLevel-1) ], 'Color',colorbase(numLevel,:),'FontSize',Fsize*1.2);

        % Add dummy patches for the legend
        p1 = patch([NaN NaN NaN NaN], [NaN NaN NaN NaN], colorbase(numLevel,:), 'EdgeColor', 'none'); %  square
        p2 = patch([NaN NaN NaN NaN], [NaN NaN NaN NaN], colorbase(7,:), 'EdgeColor', 'none'); %  square

        BR_act{1} = ['_{' modelName '}'];
        BR_act{2} = ['_{' modelName '}'];
        BR_act{3} = ['_{' modelName '}'];


        switch type
            case 'main'
                if numLevel == 3
                    legend([p1 p2 d], {'Signature of \itk + 1', 'Signature of other \itk', 'Data'}, ...
                        'Interpreter', 'tex','Location', 'northwest', 'FontSize', Fsize*1.2, 'FontWeight','normal');
                end
            case 'all'
                legend([p1 p2 d], {['Signature of level \itk\rm=' num2str(numLevel)], 'Signature of other \itk', 'Data'}, ...
                    'Interpreter', 'tex','Location', 'northwest', 'FontSize', Fsize*1.2, 'FontWeight','normal', 'Box','off');
        end


        ylim([-.5 .6])


        % Add grid for clarity
        grid off;

        % Improve aesthetics
        set(gca, 'Color', 'w','FontSize', Fsize*1.4, 'FontWeight', 'bold', 'Color','w');
        box off;

    end


    if strcmp(type,'all')
        sgtitle(['Predictive Check: ' modelName ],'FontSize',Fsize*1.5,'FontWeight','bold')
    end


    hold off



    %%

    % % Save the current figure as a .fig file
    % savefig(fig, fullfile('figures', ['PosteriorPredictive_' modelName '.fig' ]));
    % 
    % % Alternatively, to save as a high-quality PNG image:
    % saveas(gca, fullfile('figures', ['PosteriorPredictive_' modelName  '.png']));

end

if strcmp(type,'main')

    % f.Children(1:end) = [];
    sgtitle('');
    fig.Children(7).YLabel.String = '';
    fig.Children(2).Visible = 0;
    fig.Children(4).Visible = 0;
    fig.Children(6).Visible = 0;
    
    fig.Children(3).LineWidth = 1.5;
    fig.Children(5).LineWidth = 1.5;
    fig.Children(7).LineWidth = 1.5;
    
    fig.Children(3).YTickLabel = '';
    fig.Children(5).YTickLabel = '';
    
    fig.Position = [0.1885 0.3558 0.5224 0.3325];
    
    % exportgraphics(gcf,'CHASE_posterior_predictive.png');

end


