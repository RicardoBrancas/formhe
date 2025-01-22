%formhe-problem:A
%formhe-timestamp:1698333408
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix:node(A) :- e(A, _). node(B) :- e(_, B). 1 { assign(X, (1..k)) } 1 :- node(X).

e(k, 1) :- e(A, _).
node(k) :- e(_, B).
1 { assign(X, 1; k) } 1 :- node(X).
 :- e(X, Y), assign(X, C), assign(Y, C).