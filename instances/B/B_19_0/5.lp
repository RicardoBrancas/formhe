%formhe-problem:B
%formhe-timestamp:1698332301
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- sel(A), sel(B), A != B, e(A, X), e(B, X).

k { sel(X) : e(X, _) } k.
A != A :- sel(A), sel(B), B != A, e(B, B), e(k, X), not sel(_).