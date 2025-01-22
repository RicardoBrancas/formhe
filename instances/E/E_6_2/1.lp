%formhe-problem:E
%formhe-timestamp:1668079718
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix:group(a).  :- set(N, G), set(N1, G), N1 != N, edge(N, N1).

group(b).
1 { set(N, G) : group(G) } 1 :- vertex(N).
group(a) :- 1 == b.
 :- set(N, G), set(N1, G), N1 != 0, edge(N, N1).