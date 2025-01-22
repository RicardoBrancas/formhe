%formhe-problem:D
%formhe-timestamp:1667822586
%formhe-selfeval-lines:
%formhe-selfeval-fix: { sel(V) : v(V) } k.

v(X) :- e(_, X).
 :- e(V1, V2), not sel(V1), not sel(V2).
v(X) :- e(X, _).