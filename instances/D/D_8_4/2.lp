%formhe-problem:D
%formhe-timestamp:1667586197
%formhe-selfeval-lines:2
%formhe-selfeval-fix:v(X) :- e(_, X).

 :- e(V1, V2), not sel(V1), not sel(V2).
 { sel(V) : v(V) } k.
 :- e(_, X).
v(X) :- e(X, _).