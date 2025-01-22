%formhe-problem:C
%formhe-timestamp:1698333408
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:selected(X) :- sel(S), e(X, S).  :- e(X, S), not selected(X).

k { sel(S) : e(_, S) } k.
sel(X) :- selected(k), e(X, _).
 :- e(S, _), not selected(S).
-selected(0).