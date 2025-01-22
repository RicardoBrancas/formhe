%formhe-problem:D
%formhe-timestamp:1667586067
%formhe-selfeval-lines:0 2 4 5
%formhe-selfeval-fix:conected(V1, V2) :- e(V2, V1). v(X) :- e(X, _). v(X) :- e(_, X). conected(V1, V2) :- e(V1, V2).

conected(0, V2) :- e(V2, V1).
 :- conected(V1, V2), not sel(V1), not sel(V2).
v(X) :- e(X, k).
 { sel(V) : v(V) } k.
 :- e(_, X).
conected(V1, 0) :- conected(V1, V2).