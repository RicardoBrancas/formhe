%formhe-problem:C
%formhe-timestamp:1698333408
%formhe-selfeval-lines:
%formhe-selfeval-fix:k { sel(S) : e(_, S) } k.  :- e(X, S), not selected(X).

selected(X) :- sel(S), e(X, S).