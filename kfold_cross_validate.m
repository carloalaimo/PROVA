function [r_mean, r_stddev] = kfold_cross_validate(sa,na,which_features,k)

canestrell
% INPUTS : 
% sa, all features for class 1 , size : [f x Nc]
% na, " "                           [f x Nn]
% which_features, can be [4] or [4,5,14] or [-1] (to use all features)

% OUTPUTS :
% r-mean : average correct classification rate for k folds
% r-stddev : standard deviation of "..." .

if nargin<4
    k = 10;
end
if nargin<3
    which_features = -1;    
end

CIAO 

[number_of_parameters number_of_smile_samples] = size(sa); % Column-observation
[number_of_parameters number_of_nonsmile_samples ] = size(na);


smile_subsample_segments = round(linspace(1,number_of_smile_samples,k)); % indices of subsample segmentation points
nonsmile_subsample_segments = round(linspace(1,number_of_nonsmile_samples,k)); % indices of subsample segmentation points

dum = [sa,na]; % This guarantees that inputs are COLUMN-observations

if which_features ~= -1    
    sa = sa([which_features],:);
    na = na([which_features],:);    
end

% Careful  :
sa = sa';
na = na';

[number_of_smile_samples number_of_parameters ] = size(sa); % Column-observation
[number_of_nonsmile_samples number_of_parameters ] = size(na);

dum = [sa;na]; % now they should be ROW-observations.

fprintf('--------------- k-fold cr. val. : ------------- \n');
fprintf([ 'Fold: ']);

rates = [];
runtimes=[];
for i=1:k-1    
    fprintf([int2str(i) ' ']);
    
    test_s = sa(smile_subsample_segments(i):smile_subsample_segments(i+1) , :); % current_train_smiles 
    test_n = na(nonsmile_subsample_segments(i):nonsmile_subsample_segments(i+1) , :); % current_train_nonsmiles     
    
    train_s = sa;
    train_s(smile_subsample_segments(i):smile_subsample_segments(i+1),:) = [];        
    train_n = na;
    train_n(nonsmile_subsample_segments(i):nonsmile_subsample_segments(i+1),:) = [];

    [trainlen_s , num_param] = size(train_s);
    [trainlen_n , num_param] = size(train_n);
    
    % Train SVM with current training set:
    tic();    
    svmstruct = svmtrain([ones(trainlen_s,1)*1 ; ones(trainlen_n,1)*-1] , [train_s; train_n] , '-q b 0');   
    current_rate = get_classification_rate(test_s , test_n , svmstruct);
    te = toc();
    runtimes = [runtimes te];
    rates = [rates current_rate];    
    %disp(['Fold ' int2str(i) ' (of ' int2str(k) ') finished in ' num2str(te) ' seconds. Accuracy : ' num2str(round(current_rate*10000)/100) ' %']);
    
end
fprintf([ '\n']);

r_mean = mean(rates);
r_stddev = sqrt(var(rates));
  
fprintf([int2str(number_of_parameters) ' Feature(s), ' int2str(k-1) ' Folds : ' num2str(sum(runtimes)) ' s. \nAccuracy : ' num2str(r_mean*100,4) ' %%  (+-) ' num2str(r_stddev*100 , 3) ' %%.\n']);
fprintf('----------------------------------------------- \n');

end


