%formhe-problem:A
%formhe-timestamp:1667761251
%formhe-selfeval-lines:
%formhe-selfeval-fix:node(I0) :- e(_, I0). node(I0) :- e(I0, _). :- e(X,Y), assign(X,C), assign(Y,C).

% Mooshak-exercises in lab13 from ALC
% Problem A

% % Default
% #const n = 3.

% Generate
{ assign(X,1..k) } = 1 :- node(X).
% Test
% :- e(X,Y), assign(X,C), assign(Y,C).

% % Nodes
% node(1..6).
% Edges
% e(1,(2;3;4)).  e(2,(4;5;6)).  e(3,(1;4;5)).
% e(4,(1;2)).    e(5,(3;4;6)).  e(6,(2;3;5)).

% % Display
% #show color/2.