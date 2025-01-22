%formhe-problem:A
%formhe-timestamp:1
%formhe-selfeval-lines:2 3 4
%formhe-selfeval-fix:color((1..k)). 1 { assign(N, C) : color(C) } 1 :- node(N). node(N) :- e(N, _).

 :- e(N, M), assign(N, C), assign(M, C).
node(N) :- e(_, N).
color((1..0)).
1 { assign(N, C) : color(C) } 1 :- color(N).
color(N) :- e(N, C).