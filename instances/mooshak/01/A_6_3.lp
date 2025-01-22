%formhe-problem:A
%formhe-timestamp:1668077450
%formhe-selfeval-lines:3
%formhe-selfeval-fix:1 { assign(N,C) : color(C) } 1 :- node(N).

color(1..k).

node(N) :- e(_, N).
node(N) :- e(N, _).

1 { assign(N,C) } 1 :- node(N), color(C).


:- e(N, M), assign(N,C), assign(M,C).