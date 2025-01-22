%formhe-problem:D
%formhe-timestamp:1667586067
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix:conected(V1, V2) :- e(V2, V1).  :- conected(V1, V2), not sel(V1), not sel(V2). v(X) :- e(X, _).  { sel(V) : v(V) } k.

conected(1, V2) :- e(V2, V1).
 :- conected(X, V2), not sel(V1), -sel(V1).
v(V2) :- e(V2, _).
v(X) :- e(_, X).
conected(V1, V2) :- e(V1, V2).