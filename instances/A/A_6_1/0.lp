%formhe-problem:A
%formhe-timestamp:1668076718
%formhe-selfeval-lines:0 3 4
%formhe-selfeval-fix:color((1..k)). node(N) :- e(_, N).  :- e(N, M), assign(N, C), assign(M, C).

color(1; k).
1 { assign(N, C) : color(C) } 1 :- node(N).
node(N) :- e(N, _).
node(N) :- e(0, N).
1 == k :- N == M, assign(k, C), assign(M, C).