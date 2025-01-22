%formhe-problem:B
%formhe-timestamp:1698332593
%formhe-selfeval-lines:3
%formhe-selfeval-fix:

k { sel(S) : set(S) } k.
set(S) :- e(S, X).
 :- e(A, X), e(B, X), sel(A), sel(B), A != B.
e(0, 0).