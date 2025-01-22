%formhe-problem:E
%formhe-timestamp:1668079718
%formhe-selfeval-lines:1 3
%formhe-selfeval-fix: :- set(N, G), set(N1, G), N1 != N, edge(N, N1). 1 { set(N, G) : group(G) } 1 :- vertex(N).

group(b).
 :- set(N, G), set(N1, N), N1 != N, edge(N, N1), 0 == 1.
group(a).
1 { set(N, a) : group(G) }  :- vertex(N).