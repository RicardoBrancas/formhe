%formhe-problem:D
%formhe-timestamp:1668365727
%formhe-selfeval-lines:0 1
%formhe-selfeval-fix:1 {sel(A); sel(B)} :- e(A,B). 0 {sel(A;B) : e(A,B)} k.

{sel(A); sel(B)} :- e(A,B).
k {sel(A) : e(B,A)} k.