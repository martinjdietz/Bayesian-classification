
function R = sc_loo_mv_qda(Y,X,Q)

% Leave-one-out scheme for classification 
% using multivariate analysis
%
% f: R = sc_loo_mv_qda(Y,X,Q)
%
%
% Input :
% 
% Y - (n x m) training data  (n subjects, m features)          
% X - (n x p) training model (design matrix)       
% Q - (m x m) covariance basis for multivariate ReML (Q = cell(1,1))
% 
%
% Output :
%
% Posterior probability
%
% Pp - (n x k) posterior probability of classes
%
%
% Discriminant functions
% 
% L - log-likelihood + log-prior probability = ln p(x|C(k)) + ln p(C(k)) 
%
%
% References
% Bishop (2006) Pattern Recognition and Machine Learning. Springer.
% Christensen (2011) Plane Answers to Complex Questions: 
% The Theory of Linear Models. Springer.
%
% __________________________________________
% (C) 2017 Martin Dietz <martin@cfin.au.dk>


% Dependencies :
%
% MATLAB Version 8.5.0.197613 (R2015a) or higher
%
% SPM12 Version 6906 or higher
% - spm_inv.m
% - spm_logdet.m
% - spm_reml.m 
% - spm_figure.m


% dimensions

[~,p] = size(X);
[n,m] = size(Y);

if ~exist('Q','var')
    Q{1} = eye(m,m);
end 


% Leave-one-out scheme
% ---------------------------------------

for i = 1:n
    
    J    = true(n,1);
    J(i) = false;
    
    Xp = X(J,:);
    Yp = Y(J,:);


    % invert training model
    
    E = pinv(Xp'*Xp)*Xp'*Yp;
    
    
    % ReML of observation error
   
    Ce = cell(1,p);
    Le = cell(1,p);
    
    for k = 1:p
        
        Xx = Xp(logical(Xp(:,k)),k);
        Yx = Yp(logical(Xp(:,k)),1:m);
        
        Ce{k} = spm_reml(Yx'*Yx,ones(m,1),Q,length(Xx));
        Le{k} = spm_inv(Ce{k});
    end

    
    % ith subject
    
    x = Y(~J,:)';


    % prior class probabilities

    pP = sum(Xp ~= 0)/(n - 1);


    % intialise

    L = nan(1,p);

    
    % predict
    % ------------------------------------------------------------

    for k = 1:p

        % prediction error

        ey = x - E(k);


        % ln p(x|C(k)) + ln p(C(k)) 

        L(k) = - ey'*Le{k}*ey/2 + spm_logdet(Le{k})/2 + log(pP(k));
        
    end
    
    Pp = sc_softmax(L - max(L));
    
    
    % store
    
    R.Pp(i,:) = Pp;
    R.L(i,:)  = L;
    
end

Pp = R.Pp;
Pp((Pp == 0)) = max(eps(norm(Pp,'inf')),exp(-32)); 


% accuracy
% ------------------------------------

t = 0.99;

R.acc = sum(Pp(logical(X)) > t)/n;



% model performance (multivariate)
% ------------------------------------

e   = (eye(n) - X*pinv(X))*Y;
h   = diag(X*pinv(X'*X)*X');
eh  = (e./kron(ones(1,m),(1 - h)));
MSE = (trace(eh'*eh)/m);

R.mse = MSE;



% ROC curve
% ------------------------------------

% criteria

t  = 0:1e-2:1; 
Pf = Pp.*~X;


% discovery rates

tpr = nan(length(t),1);
fpr = nan(length(t),1);

for i = 1:length(t)
    tpr(i) = sum(Pp(logical(X))  > t(i))/n;  
    fpr(i) = sum(Pp(logical(~X)) > t(i))/(n*(p - 1));  
end

% R.fpr = fpr;
% R.tpr = tpr;


% graphics
% -----------------------------------

Fp = spm_figure('GetWin','Graphics');
spm_figure('Clear',Fp);
set(gcf,'Name','Bayesian quadratic discriminant analysis')


% color and font
% --------------------

c = [0 0.4470 0.7410];

fnt = 'Helvetica';
fsz = 14;

for k = 1:p
    
    % scale 
    
    if p < 4
        subplot(p + 3,1,k)
    else
        subplot(p + 1,1,k)
    end
       
    bar(Pp(:,k)), hold on
    bar(Pf(:,k),'FaceColor',c,'EdgeColor',c);
    f{k} = title(sprintf('Class %d \n',k));
    axis([0 n + 1 0 1.1])
    box off 
    
    set(gca,'FontName',fnt)
    set(gca,'FontSize',12)
    
    set(f{k},'FontSize',fsz);
    set(f{k},'FontWeight','Normal')
end

legend({'True positive' 'False positive'})

subplot(p + 1,1,p + 1)
f{1} = plot(fpr,tpr); hold on
f{2} = plot(t,t,'--k');
f{3} = title(sprintf('ROC curve \n'));

axis square

xlabel(sprintf('\n False positive rate'))
ylabel(sprintf('True positive rate \n'))

set(gca,'FontName',fnt)
set(gca,'FontSize',12)

set(f{1},'LineWidth',2);
set(f{2},'LineWidth',1);
set(f{3},'FontSize',fsz);
set(f{3},'FontWeight','Normal')




