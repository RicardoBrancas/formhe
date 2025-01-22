%formhe-problem:A
%formhe-timestamp:1668076718
%formhe-selfeval-lines:3
%formhe-selfeval-fix:node(N) :- e(_, N).

color((1..k)).
1 { assign(N, C) : color(C) } 1 :- node(N).
node(N) :- e(N, _).
node(N) :- assign(_, N).
 :- e(N, M), assign(N, C), assign(M, C).