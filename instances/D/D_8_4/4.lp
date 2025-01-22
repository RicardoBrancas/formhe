%formhe-problem:D
%formhe-timestamp:1667586197
%formhe-selfeval-lines:0 3
%formhe-selfeval-fix:v(X) :- e(_, X).  :- e(V1, V2), not sel(V1), not sel(V2).

v(X) :- e(k, X).
v(X) :- e(X, _).
 { sel(V) : v(V) } k.
 :- V1 == V2, -e(V1, V2), not sel(V2).