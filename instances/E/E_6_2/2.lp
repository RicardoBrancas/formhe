%formhe-problem:E
%formhe-timestamp:1668079718
%formhe-selfeval-lines:3
%formhe-selfeval-fix: :- set(N, G), set(N1, G), N1 != N, edge(N, N1).

group(b).
1 { set(N, G) : group(G) } 1 :- vertex(N).
group(a).
 :- set(N, G), set(N1, G), N1 != N, edge(G, N1).