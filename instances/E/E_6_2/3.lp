%formhe-problem:E
%formhe-timestamp:1668079718
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix:group(a).  :- set(N, G), set(N1, G), N1 != N, edge(N, N1).

1 { set(N, G) : group(G) } 1 :- vertex(N).
group(b).
group(b).
 :- set(N, G), edge(N1, G), a != N, edge(a, N1).