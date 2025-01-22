%formhe-problem:D
%formhe-timestamp:1698338667
%formhe-selfeval-lines:2
%formhe-selfeval-fix:1 { sel(V) : vertex(V) } k .

vertex(V) :- e(V,_).
vertex(V) :- e(_,V).

k { sel(V) : vertex(V) } k .

% Define the constraints
:- e(X,Y), not sel(X), not sel(Y).