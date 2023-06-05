% Generate
1 { set(X, a) ; set (X, b) } 1 :- vertex(X).
% Test
:- edge(X, Y), set(X, S), set(Y, S).