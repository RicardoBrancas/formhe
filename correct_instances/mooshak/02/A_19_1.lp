%formhe-problem:A
%formhe-timestamp:1698333409

% Default
%% #const k = 3.

% (Directed) Edges
%% e(a,b). e(a,c). e(b,c). e(d,c).

% Nodes
node(A) :- e(A,_).
node(B) :- e(_,B).

% Generate
{ assign(X,1..k) } = 1 :- node(X).
% Test / Integretity Constraint
:- e(X,Y), assign(X,C), assign(Y,C).

% Display
#show assign/2.