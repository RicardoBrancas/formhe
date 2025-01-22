%formhe-problem:A
%formhe-timestamp:1
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix:node(N) :- e(_, N). node(N) :- e(N, _).  :- e(N, M), assign(N, C), assign(M, C).

 :- assign(0, N), color(M; 0).
color(N) :- e(N, 1).
assign(C, (1..0)) :- e(N, M), assign(N, N), assign(M, C).
color((1..k)).
1 { assign(N, C) : color(C) } 1 :- node(N).