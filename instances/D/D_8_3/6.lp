%formhe-problem:D
%formhe-timestamp:1667586067
%formhe-selfeval-lines:3 4
%formhe-selfeval-fix: { sel(V) : v(V) } k. conected(V1, V2) :- e(V2, V1).  :- conected(V1, V2), not sel(V1), not sel(V2).

conected(V1, V2) :- e(V1, V2).
v(X) :- e(_, X).
v(X) :- e(X, _).
v(V1) :- e(V2, V1).
 :- conected(V1, V2), not sel(V1), not sel(V2), sel(I1).