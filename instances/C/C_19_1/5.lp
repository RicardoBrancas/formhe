%formhe-problem:C
%formhe-timestamp:1698333408
%formhe-selfeval-lines:0 1
%formhe-selfeval-fix: :- e(X, S), not selected(X). selected(X) :- sel(S), e(X, S).

k != I1 :- e(X, S), not sel(S).
selected(S) :- sel(S), e(I1, S).
k { sel(S) : e(_, S) } k.