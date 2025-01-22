%formhe-problem:A
%formhe-timestamp:1668076718
%formhe-selfeval-lines:0 3
%formhe-selfeval-fix:color((1..k)). 1 { assign(N, C) : color(C) } 1 :- node(N).

color(1; k).
node(N) :- e(_, N).
node(N) :- e(N, _).
1 { e(N, N) : color(C) } 1 :- node(N).
 :- e(N, M), assign(N, C), assign(M, C).