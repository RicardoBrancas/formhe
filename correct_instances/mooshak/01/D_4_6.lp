%formhe-problem:D
%formhe-timestamp:1667822587

% % Mooshak-exercises in lab13 from ALC
% % Problem D
% % constants
% #const k = 2 .

% % facts
% e(1,2) . 
% e(1,3) . 
% e(4,3) . 
% e(4,5) .

% generator
v(X):- e(_,X) .
v(X):- e(X,_) .

{ sel(V): v(V) } k .


% tester
%% stating: if vertex1 and vertex2 are connected 
%%  through an edge, then none of them must be 
%%  selected
:- e(V1, V2), not sel(V1), not sel(V2) .