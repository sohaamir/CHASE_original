function [negLL,states,data] = BAKR_2024_ToMk_model(data,model,curr_params,type)
%
%
%

%% Initialize

switch type
    
    case 'fit'

        % set game defaults, if none provided
        if ~any(isfield(data,{'strat_space','win','loss','tie'}))
            data.win = 1;
            data.loss = -1;
            data.tie = 0;
            data.strat_space = 3;
        end
        
        % extract variables
        choice_own   = data.choice_own;
        choice_other = data.choice_other;
        % nTrials  = size(choice_own,1);
        
        assert(numel(choice_own) > 1 & numel(choice_other) > 1 & numel(data.missing) > 1);  

    case 'sim'
        
        task = data;
        
        data = struct();
        [data,env] = task.fxn(data,task,'init');
        data.n_trials = numel(data.trial);
        data.missing = zeros(data.n_trials,1);
        
        [data.score_own,data.score_other,choice_own,choice_other] = deal(NaN(data.n_trials,1));
        
end
        
nTrials = data.n_trials;
nActions = data.strat_space(1);

% name parameters
params = array2table(curr_params','VariableNames',{model.params.name});

conf_opp = model.conf_opp; % specified to be 0.8 in first paper; not specified in second

% ======================== adapt to parameterization ========================= %

% % add loss aversion, if parameter provided
% if any(strcmp(params.Properties.VariableNames,'lambda')) % lossav
%     data.loss = data.loss*params.lambda;
% end

% construct payoff matrices
[pi_own, pi_other] = comp_paymatrix(data);

% set exp_max_k
% if strcmp(model.maxK_type,'fitted')
    exp_max_k = params.kappa;
% else
%     exp_max_k = model.exp_max_k; % i.e. if not provided via parameters, take from settings
% end

% ========================= pre-allocate variables =========================== %

% main variables
subj_pred = NaN(nTrials,nActions); % subject's estimated probability of the different possible opponent actions
subj_resp = NaN(nTrials,nActions); % subject's response probabilities to the estimate above (based on a noisy best-response)

subj_KL_div = NaN(nTrials,1); % belief update (BU), given by the KL-divergence between successive beliefs
subj_APE = NaN(nTrials,1); % 1 minus the probability assigned to the action chosen by the opponent (from subj_pred)
subj_SV = NaN(nTrials,1); % subjective value of choosing the different actions, given one's predictions of the opponent

% % recursive reasoning steps (either from own perspetive ('LK'), or from the perspective of the opponent ('CH-leaky'))
% if exp_max_k < 2 || strcmp(model.architecture,'LK')
%     subj_pred_k = deal(NaN(exp_max_k+1,nActions)); % 1st order beliefs (what subjects think the opponent will play)
%     subj_resp_k = deal(NaN(exp_max_k+1,nActions)); % subject's response probabilities (BR), in response to their beliefs above
% elseif exp_max_k >= 2 && contains(model.architecture,'CH')
%     opp_pred_oppk = NaN(exp_max_k,nActions); % 2nd order beliefs (what subject thinks opponent thinks about them)
%     opp_resp_oppk = NaN(exp_max_k,nActions); % opponent's response probabilities, in response to their beliefs above (as perceived by the subject)
%     subj_beliefs = ones(1,exp_max_k)/exp_max_k; % priors of the subject about the opponents's current level k, bounded by their own sophistication
% end

confidences = NaN(nTrials,exp_max_k);

% % outputs for figures
% beliefs = NaN(nTrials,3);
% likelihoods = NaN(nTrials,3);

% initialize attractions
% switch model.learning_rule
%     case 'RW-freq'
        [f_mat_own,f_mat_other] = BAKR_2024_CHASE_LR_init(model.learning_rule,nTrials,nActions); 
    % case {'RW-reward','RW-hybrid'}
    %     [f_mat_own,f_mat_other] = BAKR_2024_CHASE_LR_init(model.learning_rule,nTrials,nActions,pi_own,pi_other);
    %     score_own   = data.score_own; score_other = data.score_other;
    % case {'EWA-full','EWA-single'}
    %     [f_mat_own,f_mat_other,score_pot_own,score_pot_other,N_own,N_other] = BAKR_2024_CHASE_LR_init(...
    %                                                   model.learning_rule,nTrials,nActions,...
    %                                                   pi_own,pi_other,...
    %                                                   choice_own,choice_other);
% end

% =========================== save initial values ============================ %

f_mat_own_0 = f_mat_own(1,:);
f_mat_other_0 = f_mat_other(1,:);
% if exp_max_k >= 2 && contains(model.architecture,'CH')
%     subj_beliefs_0 = subj_beliefs;
% end
% if any(strcmp(model.learning_rule,{'EWA-full','EWA-single'}))
%     N_own_0 = N_own(1);
%     N_other_0 = N_other(1);
% end

conf = zeros(exp_max_k,1); % correct amount? correct value?

%% Trial loop
for iTrial = 1:nTrials
    
% %     if any(data.choice_own == 0 | isnan(data.choice_own))
% %         error('Missing choices - have to adapt first.');
% %     end

    % skip updates if missing trial
    if numel(data.missing) > 1 && data.missing(iTrial)
        f_mat_own(iTrial,:) = f_mat_own(max(1,iTrial-1),:);
        f_mat_other(iTrial,:) = f_mat_other(max(1,iTrial-1),:);
        % if any(strcmp(model.learning_rule,{'EWA-full','EWA-single'}))
        %     N_own(iTrial,:) = N_own(max(1,iTrial-1),:);
        %     N_other(iTrial,:) = N_other(max(1,iTrial-1),:);
        % end
        continue
    end
    
    % re-set initial values (if fitting across blocks)
    if data.trial(iTrial) == 1 && iTrial > 1
        f_mat_own_old   = f_mat_own(iTrial-1,:);
        f_mat_other_old = f_mat_other(iTrial-1,:);
        f_mat_own(iTrial-1,:)   = f_mat_own_0; % need to overwrite old ones to ensure correct update below
        f_mat_other(iTrial-1,:) = f_mat_other_0;
        % if exp_max_k >= 2
        %     subj_beliefs = subj_beliefs_0; % will be overwritten at end of trial
        % end
        % if any(strcmp(model.learning_rule,{'EWA-full','EWA-single'}))
        %     N_own_old   = N_own(iTrial-1);
        %     N_other_old = N_other(iTrial-1);
        %     N_own(iTrial-1) = N_own_0;
        %     N_other(iTrial-1) = N_other_0;
        % end
    end   
    
    % ======================================================================
    %                Confidence-weighted strategic reasoning
    % ======================================================================

    for k = 0:exp_max_k
        
        % compute relevant expected values
        if k == 0 
    
            % base for subject: responding to opponent past
            subj_EV(1,:) = pi_own*f_mat_other(max(1,iTrial-1),:)'; 
    
        else
    
            % ------------------------- opponent model ---------------------- %
            if k == 1 
    
                % base for opp: responding to subj past (<- reference for even subj levels)
                opp_EV(1,:) = pi_other*f_mat_own(max(1,iTrial-1),:)'; 
    
            else
        
                if k == 2 
    
                    % responding to opp prediction for one's own k0 behaviour (<- reference for odd subj levels)
                    opp_subj_EV(1,:) = subj_EV(1,:); 
            
                else % all higher levels (should be self-consistent, as all based on argmax and conf_opp)
    
                    % presumed subj response to two levels lower
                    opp_subj_pred_k(k-2,:) = argmax_fxn(opp_EV(k-2,:)); % argmax
                    opp_subj_EV(k-1,:) = conf_opp * (pi_own*opp_subj_pred_k(k-2,:)') + (1-conf_opp) * opp_subj_EV(k-2,:)';
    
                end
    
                % opp response to predicted subj action
                opp_pred_k(k-1,:) = argmax_fxn(opp_subj_EV(k-1,:));
                opp_EV(k,:) = conf_opp * (pi_other*opp_pred_k(k-1,:)') + (1-conf_opp) * opp_EV(k-1,:)';
        
            end
            % ------------------------- /opponent model --------------------- %
    
            % integrate response to opp prediction into expected value
            subj_pred_k(k,:) = argmax_fxn(opp_EV(k,:)); % argmax
            subj_EV(k+1,:) = conf(k) * (pi_own*subj_pred_k(k,:)') + (1-conf(k)) * subj_EV(k,:)';
    
        end
    
        % add noise
        subj_resp_k(k+1,:) = softmax_fxn(subj_EV(k+1,:),params.beta);
    
        % % save
        % subj_pred(iTrial,:) = subj_pred_k(end,:);
        % subj_resp(iTrial,:) = subj_resp_k(end,:);
    
    end

    % subject's weighted opponent action prediction & response to that
    if exp_max_k > 0
        subj_pred(iTrial,:) = subj_pred_k(end,:);
    end
    subj_resp(iTrial,:) = subj_resp_k(end,:);
    
    % =========================== Action simulation ========================== %
    
    if strcmp(type,'sim')
        
        % subject
        choice_own(iTrial) = find(mnrnd(1,subj_resp(iTrial,:)));
        data.choice_own(iTrial) = choice_own(iTrial);

        % bot
        [data,env] = task.fxn(data,task,'update',env,iTrial); % need only: choice_other(iTrial)
        choice_other(iTrial,1) = data.choice_other(iTrial);
        score_own(iTrial,1)    = data.score_own(iTrial);
        score_other(iTrial,1)  = data.score_other(iTrial);

    end
    
    % ======================= confidence updates ======================== %

    if exp_max_k > 0

        % compare predictions to observed choice
        [~,pred] = max(subj_pred_k,[],2);
        success = double(pred == choice_other(iTrial));
    
        % weed out successes that are already explained by a lower level
        if sum(success) > 1
            explained = find(success);
            success(explained(2:end)) = 2;
        end
    
        for k = 1:exp_max_k
    
            switch success(k)
                case 0, conf(k) = (1-params.alpha) * conf(k);                % wrong: only decay
                case 1, conf(k) = (1-params.alpha) * conf(k) + params.alpha; % correct: full update
                case 2, conf(k) = conf(k);                                   % already predicted: no change
            end
    
        end

        confidences(iTrial,:) = conf;
    
        % compute internal states
        subj_APE(iTrial) = 1 - subj_pred(iTrial,choice_other(iTrial));
        subj_SV(iTrial) = pi_own(choice_own(iTrial),:) * subj_pred(iTrial,:)';
    
    end

    % ==========================================================================
    %                             Level-0 update
    % ==========================================================================
    
    % L0 representation
    % switch model.learning_rule
    %     case 'RW-freq'
            [f_mat_own,f_mat_other] = BAKR_2024_CHASE_LR_update(model.learning_rule,iTrial,f_mat_own,f_mat_other,...
                                                        choice_own,choice_other,params);
        % case {'RW-reward','RW-hybrid'}
        %     [f_mat_own,f_mat_other] = BAKR_2024_CHASE_LR_update(model.learning_rule,iTrial,f_mat_own,f_mat_other,...
        %                                                 choice_own,choice_other,params,...
        %                                                 score_own,score_other);
        % case {'EWA-full','EWA-single'}
        %     [f_mat_own,f_mat_other,N_own,N_other] = BAKR_2024_CHASE_LR_update(model.learning_rule,iTrial,f_mat_own,f_mat_other,...
        %                                                 choice_own,choice_other,params,...
        %                                                 score_pot_own,score_pot_other,N_own,N_other);
    % end

    % put overwritten values back (if fitting across blocks) <- cannot do earlier because this is used for update above
    if data.trial(iTrial) == 1 && iTrial > 1
        f_mat_own(iTrial-1,:) = f_mat_own_old;
        f_mat_other(iTrial-1,:) = f_mat_other_old;
        % if any(strcmp(model.learning_rule,{'EWA-full','EWA-single'}))
        %     N_own(iTrial-1) = N_own_old;
        %     N_other(iTrial-1) = N_other_old;
        % end
    end

end

%% Post-processing

% internal states
states.subj_SV = subj_SV;
states.subj_APE = subj_APE;
states.subj_KL_div = subj_KL_div;

% % states for figures
% states.f_mat_own  = f_mat_own;
% states.likelihoods = likelihoods;
% states.beliefs = beliefs;

states.beliefs = confidences;

switch type

    case 'fit'

        lik = NaN(nTrials,1);
        lik(~data.missing) = subj_resp(sub2ind([nTrials,nActions],find(~data.missing),choice_own(~data.missing))); % chosen action
        states.lik = lik;
        
        negLL = -sum(log(lik(~data.missing)));

    case 'sim'

        states.level_k = env.level_k;

        states.score = NaN(nTrials,1);
        for i_block = 1:numel(task.bot.levels)
            idx = (40*(i_block-1))+1:(40*(i_block));
            states.score(idx) = cumsum(score_own(idx));
        end

        data.choice_own = choice_own;
        data.choice_other = choice_other;
        data.score_own = score_own;
        data.score_other = score_other;

        data.n_blocks = 1;

        data.bot_level = task.bot.levels;
        
        negLL = NaN;

end

end

function p_actions = softmax_fxn(inputs,beta)

p_actions = exp(beta*inputs)/(sum(exp(beta*inputs)));

end

function a = argmax_fxn(EV)

a = zeros(size(EV));
[~,idx] = max(EV);
a(idx) = 1;

end
