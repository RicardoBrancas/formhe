%formhe-problem:D
%formhe-timestamp:1667822586
%formhe-selfeval-lines:2
%formhe-selfeval-fix: :- e(V1, V2), not sel(V1), not sel(V2).

v(X) :- e(_, X).
 { sel(V) : v(V) } k.
 :- e(V1, V2), not sel(_), not e(V2, 1; 1).
v(X) :- e(X, _).