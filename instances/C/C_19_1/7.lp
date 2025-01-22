%formhe-problem:C
%formhe-timestamp:1698333408
%formhe-selfeval-lines:1
%formhe-selfeval-fix:selected(X) :- sel(S), e(X, S).

k { sel(S) : e(_, S) } k.
selected(X) :- sel(I0), e(X, S).
 :- e(X, S), not selected(X).