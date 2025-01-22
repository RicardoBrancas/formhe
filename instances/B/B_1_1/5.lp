%formhe-problem:B
%formhe-timestamp:1668364658
%formhe-selfeval-lines:0
%formhe-selfeval-fix:-sel(A) :- e(A, Y), e(B, Y), sel(B), A != B.

-sel(A) :- e(A, I1), B == Y, sel(B), A == B.
k { sel(A) : e(A, Y) } k.