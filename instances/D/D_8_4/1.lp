%formhe-problem:D
%formhe-timestamp:1667586197
%formhe-selfeval-lines:0
%formhe-selfeval-fix:v(X) :- e(_, X).

v(k) :- e(_, X).
v(X) :- e(X, _).
 { sel(V) : v(V) } k.
 :- e(V1, V2), not sel(V1), not sel(V2).