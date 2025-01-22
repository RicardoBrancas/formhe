%formhe-problem:A
%formhe-timestamp:1667929699
%formhe-selfeval-lines:0 1
%formhe-selfeval-fix:1 { assign(X, (1..k)) } 1 :- e(X, Y). 1 { assign(Y, (1..k)) } 1 :- e(X, Y).

1 { assign(X, (0..0)) } 1 :- e(X, Y).
 :- e(X, Y).
 :- e(X, Y), assign(X, C), assign(Y, C).