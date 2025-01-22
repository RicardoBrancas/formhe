%formhe-problem:D
%formhe-timestamp:1667822586
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:v(X) :- e(X, _).  { sel(V) : v(V) } k.  :- e(V1, V2), not sel(V1), not sel(V2).

v(X) :- e(_, X).
sel(X) :- e(X, _).
 { sel(V) : v(V) } 0.
 :- e(V1, V2), not sel(V1), not v(V2).