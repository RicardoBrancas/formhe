%formhe-problem:D
%formhe-timestamp:1667586197
%formhe-selfeval-lines:1
%formhe-selfeval-fix: { sel(V) : v(V) } k.

 :- e(V1, V2), not sel(V1), not sel(V2).
 { v(V) : e(V, I1) } 1.
v(X) :- e(_, X).
v(X) :- e(X, _).