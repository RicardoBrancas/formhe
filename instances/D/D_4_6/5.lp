%formhe-problem:D
%formhe-timestamp:1667822586
%formhe-selfeval-lines:1 2 4
%formhe-selfeval-fix:v(X) :- e(_, X).  :- e(V1, V2), not sel(V1), not sel(V2).

v(X) :- e(X, _).
 :- e(_, _).
 :- V1 == V2, not sel(V1), -sel(V2).
 { sel(V) : v(V) } k.
 :- sel(V).