%formhe-problem:A
%formhe-timestamp:1668076718
%formhe-selfeval-lines:0 3 4
%formhe-selfeval-fix:color((1..k)). 1 { assign(N, C) : color(C) } 1 :- node(N).  :- e(N, M), assign(N, C), assign(M, C).

color(1; k).
node(N) :- e(_, N).
node(N) :- e(N, _).
1 { assign(1, C) : e(k, C) } 1 :- node(N).
 :- e(N, I0), N == C, assign(_, C).