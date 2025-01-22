%formhe-problem:D
%formhe-timestamp:1667585760
%formhe-selfeval-lines:2
%formhe-selfeval-fix:0 { sel(I0) : v(I0) } k.

v(X):- e(_,X).
v(X):- e(X,_).

{ sel(V): v(V)} n.

conected(V1, V2):- e(V1, V2).
conected(V1, V2):- e(V2, V1).



:- conected(V1, V2), not sel(V1), not sel(V2).