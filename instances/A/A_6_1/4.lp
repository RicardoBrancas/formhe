%formhe-problem:A
%formhe-timestamp:1668076718
%formhe-selfeval-lines:1 4
%formhe-selfeval-fix:1 { assign(N, C) : color(C) } 1 :- node(N).  :- e(N, M), assign(N, C), assign(M, C).

color((1..k)).
1 { assign(N, C) : color(C) } 1 :- node(N), M == k.
node(N) :- e(N, _).
node(N) :- e(_, N).
 :- e(0, M), e(N, C), e(M, C).