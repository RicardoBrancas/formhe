%formhe-problem:A
%formhe-timestamp:1668076718
%formhe-selfeval-lines:4
%formhe-selfeval-fix: :- e(N, M), assign(N, C), assign(M, C).

node(N) :- e(_, N).
1 { assign(N, C) : color(C) } 1 :- node(N).
color((1..k)).
node(N) :- e(N, _).
 :- e(N, I1), assign(N, C), assign(M, C), e(_, _).