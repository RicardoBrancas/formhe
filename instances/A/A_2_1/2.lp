%formhe-problem:A
%formhe-timestamp:1667929699
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix: :- e(X, Y), assign(X, C), assign(Y, C). 1 { assign(Y, (1..k)) } 1 :- e(X, Y).

1 { assign(X, (1..k)) } 1 :- e(X, Y).
Y != C :- assign(X, Y), assign(X, Y), assign(X, C).
1 { e(Y, 1; k) } 1 :- e(X, Y).