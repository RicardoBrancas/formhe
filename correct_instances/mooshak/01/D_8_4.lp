%formhe-problem:D
%formhe-timestamp:1667586198

v(X):- e(_,X).
v(X):- e(X,_).

{ sel(V): v(V) } k.



:- e(V1, V2), not sel(V1), not sel(V2).