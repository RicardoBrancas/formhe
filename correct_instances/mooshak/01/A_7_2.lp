%formhe-problem:A
%formhe-timestamp:1667731438

% K-coloring problem
color(1..k).


vertex(X) :- e(X,_).
vertex(X) :- e(_,X).

% Generation

%% Assign one color to each vertex

1{assign(X,C) : color(C)}1 :- vertex(X).

% Test

:- e(X, Y), assign(X,C), assign(Y,C) , color(C).
:- e(Y, X), assign(X,C), assign(Y,C) , color(C).