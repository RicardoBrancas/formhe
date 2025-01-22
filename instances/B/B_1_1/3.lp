%formhe-problem:B
%formhe-timestamp:1668364658
%formhe-selfeval-lines:0 2
%formhe-selfeval-fix:-sel(A) :- e(A, Y), e(B, Y), sel(B), A != B.

-sel(A) :- e(B, A), e(B, Y), sel(A), A == I1.
k { sel(A) : e(A, Y) } k.
 :- not sel(0).