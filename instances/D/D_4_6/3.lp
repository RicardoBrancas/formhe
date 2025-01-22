%formhe-problem:D
%formhe-timestamp:1667822586
%formhe-selfeval-lines:3
%formhe-selfeval-fix:v(X) :- e(_, X).

 :- e(V1, V2), not sel(V1), not sel(V2).
 { sel(V) : v(V) } k.
v(X) :- e(X, _).
v(X) :- e(I1, X), e(_, 0).