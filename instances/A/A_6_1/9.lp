%formhe-problem:A
%formhe-timestamp:1668076718
%formhe-selfeval-lines:3 5
%formhe-selfeval-fix: :- e(N, M), assign(N, C), assign(M, C).

color((1..k)).
1 { assign(N, C) : color(C) } 1 :- node(N).
node(N) :- e(N, _).
 :- assign(N, M), assign(N, C), assign(M, C).
node(N) :- e(_, N).
 :- color(_).