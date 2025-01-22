%formhe-problem:A
%formhe-timestamp:1698332301
%formhe-selfeval-lines:3
%formhe-selfeval-fix: :- e(X, Y), assign(X, C), assign(Y, C).

node(B) :- e(_, B).
node(A) :- e(A, _).
1 { assign(X, (1..k)) } 1 :- node(X).
 :- e(X, Y), assign(X, C), assign(C, C).