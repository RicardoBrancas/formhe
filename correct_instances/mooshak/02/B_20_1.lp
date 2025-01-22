%formhe-problem:B
%formhe-timestamp:1698332594

set(S) :- e(S,X).
:- e(A,X), e(B,X), sel(A), sel(B), A != B.
k{ sel(S) : set(S) }k.
#show sel/1.