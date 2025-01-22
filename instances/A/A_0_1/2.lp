%formhe-problem:A
%formhe-timestamp:1
%formhe-selfeval-lines:3 4
%formhe-selfeval-fix:1 { assign(N, C) : color(C) } 1 :- node(N).  :- e(N, M), assign(N, C), assign(M, C).

color((1..k)).
node(N) :- e(_, N).
node(N) :- e(N, _).
 :- node(N).
 :- e(N, M), assign(N, C), assign(C, C).