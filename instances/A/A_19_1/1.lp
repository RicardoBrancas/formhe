%formhe-problem:A
%formhe-timestamp:1698333408
%formhe-selfeval-lines:0 2 3
%formhe-selfeval-fix: :- e(X, Y), assign(X, C), assign(Y, C). 1 { assign(X, (1..k)) } 1 :- node(X). node(B) :- e(_, B).

 :- e(X, Y), assign(X, C), Y != C, node(0).
node(A) :- e(A, _).
1 { assign(X, (0..k)) } k :- node(X).
node(0) :- 0 == B.