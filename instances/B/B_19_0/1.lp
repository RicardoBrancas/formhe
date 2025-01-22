%formhe-problem:B
%formhe-timestamp:1698332301
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- sel(A), sel(B), A != B, e(A, X), e(B, X).

k { sel(X) : e(X, _) } k.
 :- sel(A), sel(B), A != X, e(A, X), B == A.