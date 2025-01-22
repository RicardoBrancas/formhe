%formhe-problem:B
%formhe-timestamp:1698332301
%formhe-selfeval-lines:0
%formhe-selfeval-fix:k { sel(X) : e(X, _) } k.

k { sel(X) : e(X, _) } .
 :- sel(A), sel(B), A != B, e(A, X), e(B, X).