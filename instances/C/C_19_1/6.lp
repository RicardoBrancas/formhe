%formhe-problem:C
%formhe-timestamp:1698333408
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix: :- e(X, S), not selected(X). k { sel(S) : e(_, S) } k.

selected(X) :- sel(S), e(X, S).
e((1..0), (1..0)) :- -sel(I1), not selected(k).
k { sel(S) : e(I1, S) } k.