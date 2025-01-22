%formhe-problem:D
%formhe-timestamp:1667822586
%formhe-selfeval-lines:0 3
%formhe-selfeval-fix:v(X) :- e(_, X).  :- e(V1, V2), not sel(V1), not sel(V2).

v(1) :- e(_, X).
v(X) :- e(X, _).
 { sel(V) : v(V) } k.
 :- V1 == V2, -sel(V2), not v(V1).