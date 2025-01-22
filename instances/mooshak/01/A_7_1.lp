%formhe-problem:A
%formhe-timestamp:1667730943

% K-coloring problem
color(1..k)

%% If it appears in the edge relation then it is a vertex

vertex(X) :- edge(X,_)
vertex(X) :- edge(_,X)

% Generation

%% Assign one color to each vertex

1{assign(X,C) : color(C)}1 :- vertex(X)

% Test

:- edge(X, Y), assign(X,C), assign(Y,C) , color(C)