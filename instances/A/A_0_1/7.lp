%formhe-problem:A
%formhe-timestamp:1
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:node(N) :- e(N, _). 1 { assign(N, C) : color(C) } 1 :- node(N).  :- e(N, M), assign(N, C), assign(M, C).

node(N) :- e(_, N).
node(N) :- e(N, _), node(0; 0).
1 { e(N, C) : color(C) } 1 :- node(N).
 :- e(N, M), M == C, assign(C, C), e((1..0), 0; 0).
color((1..k)).