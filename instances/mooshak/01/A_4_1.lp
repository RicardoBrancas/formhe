%formhe-problem:A
%formhe-timestamp:1667760984
%formhe-selfeval-lines:
%formhe-selfeval-fix:node(I0) :- e(_, I0). node(I0) :- e(I0, _).

% Mooshak-exercises in lab13 from ALC
% Problem A

% % Default
% #const n = 3.

% Generate
{ assign(X,1..k) } = 1 :- node(X).
% Test
:- e(X,Y), assign(X,C), assign(Y,C).

% % Nodes
% node(1..6).
% Edges


% % Display
% #show color/2.