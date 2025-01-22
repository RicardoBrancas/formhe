%formhe-problem:E
%formhe-timestamp:1668079719

group(a).
group(b).


1 { set(N,G): group(G) } 1 :- vertex(N).


:- set(N, G), set(N1, G), N1 != N, edge(N, N1).