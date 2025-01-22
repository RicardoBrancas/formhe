%formhe-problem:A
%formhe-timestamp:1668076718
%formhe-selfeval-lines:1 3
%formhe-selfeval-fix:1 { assign(N, C) : color(C) } 1 :- node(N). node(N) :- e(N, _).

node(N) :- e(_, N).
1 { assign(0, C) : color(C) } k :- node(N), assign(0; 0, 1; k).
color((1..k)).
node(N) :- e(N, _), 1 == 0.
 :- e(N, M), assign(N, C), assign(M, C).