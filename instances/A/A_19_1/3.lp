%formhe-problem:A
%formhe-timestamp:1698333408
%formhe-selfeval-lines:0 2 3
%formhe-selfeval-fix:node(B) :- e(_, B). node(A) :- e(A, _).  :- e(X, Y), assign(X, C), assign(Y, C).

node(B) :- e(I0, B).
1 { assign(X, (1..k)) } 1 :- node(X).
 :- e(I1, _).
1 == k :- e(0, Y), assign(C, C), Y == C.