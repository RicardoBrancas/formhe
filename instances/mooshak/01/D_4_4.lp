%formhe-problem:D
%formhe-timestamp:1667788621

% % Mooshak-exercises in lab13 from ALC
% % Problem D
% % constants
% % #const k = 2 .

% % facts
% e(1,2). 
% e(1,3). 
% e(4,3). 
% e(4,5).

% limit(k).
% generator
c(k)

sel(X) :- e(X, Y), not sel(Y) .
solve :- limit(c), #count{ vertex(X) : sel(X) } <= c .


% tester
:- not solve .