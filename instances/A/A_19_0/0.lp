%formhe-problem:A
%formhe-timestamp:1698332301
%formhe-selfeval-lines:
%formhe-selfeval-fix:node(B) :- e(_, B).  :- e(X, Y), assign(X, C), assign(Y, C).

1 { assign(X, (1..k)) } 1 :- node(X).
node(A) :- e(A, _).