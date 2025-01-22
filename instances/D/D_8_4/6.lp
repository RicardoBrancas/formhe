%formhe-problem:D
%formhe-timestamp:1667586197
%formhe-selfeval-lines:2
%formhe-selfeval-fix: { sel(V) : v(V) } k.  :- e(V1, V2), not sel(V1), not sel(V2).

v(X) :- e(_, X).
v(X) :- e(X, _).
 :- e(V1, V2), -sel(V1), not sel(V2).