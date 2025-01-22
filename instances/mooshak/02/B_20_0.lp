%formhe-problem:B
%formhe-timestamp:1698332122
%formhe-selfeval-lines:0 1
%formhe-selfeval-fix:set(S) :- e(S,X). :- e(A,X), e(B,X), sel(A), sel(B), A != B.

set(S) :- e(X,S).
:- e(A,X), e(B,X), sel(A), sel(B).
k{ sel(X) : set(X) }k.
#show sel/1.