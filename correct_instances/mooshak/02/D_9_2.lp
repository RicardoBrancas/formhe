%formhe-problem:D
%formhe-timestamp:1698339289

vertex(V) :- e(V,_).
vertex(V) :- e(_,V).

1 { sel(V) : vertex(V) } k .

% Define the constraints
:- e(X,Y), not sel(X), not sel(Y).