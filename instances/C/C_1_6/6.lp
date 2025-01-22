%formhe-problem:C
%formhe-timestamp:1668360092
%formhe-selfeval-lines:0
%formhe-selfeval-fix:-sel(B) :- e(Y, A), e(Y, B), sel(A), A != B.

-sel(B) :- A != 0, e(Y, B), sel(A), A != B.
k { sel(A) : e(Y, A) } k.