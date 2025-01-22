%formhe-problem:C
%formhe-timestamp:1668360092
%formhe-selfeval-lines:1
%formhe-selfeval-fix:k { sel(A) : e(Y, A) } k.

-sel(B) :- e(Y, A), e(Y, B), sel(A), A != B.
k { sel(Y) : e(B, Y) } 0 :- e((1..0), k; 0).