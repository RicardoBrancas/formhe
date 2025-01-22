%formhe-problem:D
%formhe-timestamp:1667822586
%formhe-selfeval-lines:0 2 3
%formhe-selfeval-fix: :- e(V1, V2), not sel(V1), not sel(V2). v(X) :- e(X, _). v(X) :- e(_, X).

 :- e(V1, V2), not sel(V1), not e(V2, _).
 { sel(V) : v(V) } k.
v(X) :- e(X, _), e(_, V1).
v(X) :- e(_, 0), 0 == X.