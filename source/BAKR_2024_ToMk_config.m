function settings = BAKR_2024_CHASE_config()
% 
%
%

% Niklas Buergi, 2024

%% model specification
S.name       = sprintf('ToMk');
S.loglik_fxn = @BAKR_2024_ToMk_model;

%% fitting

S.conf_opp = 0.8;
S.learning_rule = 'RW-freq';

% k is treated as different models in their paper; but we should treat it
% as parameter to make it comparable to CHASE

n_samples = 4;

params(1).name    = 'alpha'; % called lambda in their work
params(1).support = [0,1];
params(1).grid    = linspace(0.3,0.99,n_samples); % <- might consider different range

params(2).name    = 'beta';
params(2).support = [0,10];
params(2).grid    = linspace(0.1,5,n_samples); 

params(3).name    = 'kappa'; % maxK
params(3).support = [0,3];
params(3).space   = 'integer';

params = mn_createParamTable(params,n_samples);
params = table2struct(params)';

%% out
S.params = params;
settings = S;

end
