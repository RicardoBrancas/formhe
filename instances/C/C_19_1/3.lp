%formhe-problem:C
%formhe-timestamp:1698333408
%formhe-selfeval-lines:0
%formhe-selfeval-fix: :- e(X, S), not selected(X).

 :- e(X, k), not selected(X).
selected(X) :- sel(S), e(X, S).
k { sel(S) : e(_, S) } k.