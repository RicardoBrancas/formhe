%formhe-problem:C
%formhe-timestamp:1698333408
%formhe-selfeval-lines:0 1
%formhe-selfeval-fix:selected(X) :- sel(S), e(X, S).  :- e(X, S), not selected(X).

 :- selected(S), e(0, I1).
 :- e(X, S), not sel(X), e(I0; k, k; S).
k { sel(S) : e(_, S) } k.