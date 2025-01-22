%formhe-problem:D
%formhe-timestamp:1667586197
%formhe-selfeval-lines:2 3 4
%formhe-selfeval-fix:v(X) :- e(X, _).  :- e(V1, V2), not sel(V1), not sel(V2).

v(X) :- e(_, X).
 { sel(V) : v(V) } k.
v(I1) :- e(I1, _).
 :- e(V1, V2), not sel(V2), -e(V1, V2).
 :- 1 == 0.