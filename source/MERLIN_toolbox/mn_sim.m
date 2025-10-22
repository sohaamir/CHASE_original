function sim = mn_sim(task,model,params)
%
% possible inputs:
%  - environment function
%  - inputs (i.e. stimuli that the agent receives
%  - some kind of settings?
%  - ...
%

% cave: if simulating interactions, first model will be default model to
% define simulation function

subj = struct();

% if parameters are provided, just return a single evaluation
if nargin > 2 && ~isempty(params)
    
    % check number of parameters
    error_msg = 'Wrong number of parameters provided (%i/%i).';
    
    if isa(params,'double')
        % Check parameter count
        assert(numel(params) == numel(model.params), error_msg, numel(params), numel(model.params));
        
        % Ensure params is a COLUMN vector for model call
        params = params(:);  % Force to column (5×1)
        
        % Call model with column vector (model expects this)
        [~,subj.states,subj.data] = model.sim_fxn(task, model, params, 'sim');
        
    elseif isa(params,'cell') % for interactive simulations
        % Check parameter count for each agent
        for ii = 1:numel(params)
            assert(numel(params{ii}) == numel(model{ii}.params), error_msg, numel(params{ii}), numel(model{ii}.params));
        end
        
        % Setup for multi-agent simulation
        agents = model;
        params_per_agent = params;
        model = model{1};
        params = params{1};
        
        % Ensure params is a COLUMN vector
        params = params(:);
        
        % Call model for interactive simulation
        [~,subj.states,subj.data] = model.sim_fxn(task, agents, params_per_agent, 'sim');
    end
    
    % Create table from ROW vector (array2table expects this)
    subj.params = array2table(params', 'VariableNames', {model.params.name});
    
    sim.type   = 'sim';
    sim.task   = task;
    sim.model  = model;
    sim.params = params;
    sim.subj   = subj;
    sim.n      = 1;
    
    return
    
end

% rename
config = mn_sim_config();
nParams = size(model.params,2);

% create parameter combinations
switch config.type
    
    case 'grid'
        
        [paramCombs,nCombs] = mn_getGridMat(model.params.grid);
        
    case {'prior','uniform'}
        
        nCombs = config.max_combinations;
        switch config.type
            case 'prior',   sampling_fxn = @mn_samplePrior;
            case 'uniform', sampling_fxn = @mn_sampleUniform;
        end
        for iParam = 1:nParams     
            paramCombs(:,iParam) = sampling_fxn(model,iParam,nCombs);
        end   
        
end

% simulate all
for iComb = 1:nCombs
    mn_printProgress(iComb,nCombs,'Simulating... ');
    curr_params = paramCombs(iComb,:);
    [~,subj(iComb).states,subj(iComb).data] = model.sim_fxn(task,model,curr_params','sim');
end

% pack
sim.type   = 'sim';
sim.task   = task;
sim.model  = model;
sim.config = config;
sim.params = paramCombs;
sim.subj   = subj';
sim.n      = nCombs;

end
