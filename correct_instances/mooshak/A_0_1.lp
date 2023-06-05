color(1..k).
node(N) :- e(_, N).
node(N) :- e(N, _).
1 { assign(N,C) : color(C) } 1 :- node(N).
:- e(N,M), assign(N,C), assign(M,C).