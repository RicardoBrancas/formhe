%formhe-problem:B
%formhe-timestamp:1698332593
%formhe-selfeval-lines:0
%formhe-selfeval-fix: :- e(A, X), e(B, X), sel(A), sel(B), A != B.

 :- e(A, X), e(B, X), set(A), sel(B), e(A, B).
set(S) :- e(S, X).
k { sel(S) : set(S) } k.