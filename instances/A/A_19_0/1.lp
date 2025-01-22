%formhe-problem:A
%formhe-timestamp:1698332301
%formhe-selfeval-lines:4
%formhe-selfeval-fix:

node(A) :- e(A, _).
node(B) :- e(_, B).
1 { assign(X, (1..k)) } 1 :- node(X).
 :- e(X, Y), assign(X, C), assign(Y, C).
 :- node((0..k)).