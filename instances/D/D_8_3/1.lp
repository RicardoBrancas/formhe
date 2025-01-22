%formhe-problem:D
%formhe-timestamp:1667586067
%formhe-selfeval-lines:1 3 5
%formhe-selfeval-fix:v(X) :- e(X, _). conected(V1, V2) :- e(V1, V2).  :- conected(V1, V2), not sel(V1), not sel(V2).

v(X) :- e(_, X).
v(X) :- e(X, _), 0 == k.
 { sel(V) : v(V) } k.
 :- e(V1, V1).
conected(V1, V2) :- e(V2, V1).
e((k..0), (k..0)) :- conected(V1, V2), not sel(V1), not sel(V2).