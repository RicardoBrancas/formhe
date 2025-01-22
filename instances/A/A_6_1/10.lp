%formhe-problem:A
%formhe-timestamp:1668076718
%formhe-selfeval-lines:2 4
%formhe-selfeval-fix:color((1..k)).  :- e(N, M), assign(N, C), assign(M, C).

node(N) :- e(_, N).
1 { assign(N, C) : color(C) } 1 :- node(N).
color(1; k).
node(N) :- e(N, _).
 :- N == M, k != C, assign(M, C).