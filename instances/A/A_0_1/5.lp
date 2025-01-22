%formhe-problem:A
%formhe-timestamp:1
%formhe-selfeval-lines:1 3 4
%formhe-selfeval-fix:node(N) :- e(_, N). 1 { assign(N, C) : color(C) } 1 :- node(N). node(N) :- e(N, _).

 :- e(N, M), assign(N, C), assign(M, C).
 :- e(_, N).
color((1..k)).
1 { assign(N, C) : color(C) } 1 :- color(N).
N != C :- e(N, C), N == I1.