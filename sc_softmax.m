
function P = sc_softmax(x,k)

% Softmax function
% f: P = vb_softmax(x,k)
% 
% x  - values or log-evidences
% k  - sensitivity parameter (default = 1)
% 
% P  - probabilities - exp(x*k)/sum(exp(x*k))
% 
% References
% Bishop (2006) Pattern Recognition and Machine Learning. Springer
%
% __________________________________________
% (C) 2016 Martin Dietz <martin@cfin.au.dk>


if nargin > 1
    x = x*k;
end

 P = exp(x)/sum(exp(x));

