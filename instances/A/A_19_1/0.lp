%formhe-problem:A
%formhe-timestamp:1698333408
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix:node(A) :- e(A, _).  :- e(X, Y), assign(X, C), assign(Y, C). node(B) :- e(_, B).

node(A) :- e(A, k).
assign((k..1), (k..I1)) :- e(X, Y), X == C, Y != C.
node(B) :- assign(_, B).
1 { assign(X, (1..k)) } 1 :- node(X).