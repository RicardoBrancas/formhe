%formhe-problem:C
%formhe-timestamp:1668360092
%formhe-selfeval-lines:1
%formhe-selfeval-fix:-sel(B) :- e(Y, A), e(Y, B), sel(A), A != B.

k { sel(A) : e(Y, A) } k.
-sel(B) :- Y == A, e(Y, B), sel(A), A != B.