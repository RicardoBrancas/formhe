%formhe-problem:B
%formhe-timestamp:1668364658
%formhe-selfeval-lines:1
%formhe-selfeval-fix:-sel(A) :- e(A, Y), e(B, Y), sel(B), A != B.

k { sel(A) : e(A, Y) } k.
not sel(A) :- B != Y, e(B, Y), B != 0, e(A, I0), B == k.