%formhe-problem:D
%formhe-timestamp:1667788815
%formhe-selfeval-lines:0 1 2 3
%formhe-selfeval-fix:{ sel(X;Y) : e(X,Y) } k. :- not sel(X), not sel(Y), e(X,Y).

% Mooshak-exercises in lab13 from ALC
% Problem D
% constants
% #const k = 2 .

% % facts
% e(1,2) . 
% e(1,3) . 
% e(4,3) . 
% e(4,5) .

limit(k) .

% generator

sel(X) :- e(X, Y), not sel(Y) .
solve :- limit(C), #count{ vertex(X) : sel(X) } <= C .


% tester
:- not solve .