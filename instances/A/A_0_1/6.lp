%formhe-problem:A
%formhe-timestamp:1
%formhe-selfeval-lines:3 4 5
%formhe-selfeval-fix: :- e(N, M), assign(N, C), assign(M, C). color((1..k)).

node(N) :- e(_, N).
node(N) :- e(N, _).
1 { assign(N, C) : color(C) } 1 :- node(N).
 :- C != M, e(N, C), assign(M, M).
color((0..1)) :- not color(1).
assign(k, k).