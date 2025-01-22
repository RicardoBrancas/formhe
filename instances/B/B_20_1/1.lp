%formhe-problem:B
%formhe-timestamp:1698332593
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix:k { sel(S) : set(S) } k.  :- e(A, X), e(B, X), sel(A), sel(B), A != B.

set(S) :- e(S, X).
 { sel(S) : set(S) } .
 :- A == X, B == X, sel(A), set(A), 1 != B.