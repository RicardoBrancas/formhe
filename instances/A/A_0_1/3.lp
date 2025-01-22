%formhe-problem:A
%formhe-timestamp:1
%formhe-selfeval-lines:3
%formhe-selfeval-fix:1 { assign(N, C) : color(C) } 1 :- node(N).

 :- e(N, M), assign(N, C), assign(M, C).
node(N) :- e(_, N).
color((1..k)).
1 { e(N, C) : color(C) } 1 :- node(N).
node(N) :- e(N, _).