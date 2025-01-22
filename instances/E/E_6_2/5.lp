%formhe-problem:E
%formhe-timestamp:1668079718
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix: :- set(N, G), set(N1, G), N1 != N, edge(N, N1). group(a).

group(b).
 :- N != G, set(N1, G), N1 != N, edge(N, N1).
vertex(a) :- edge(G; 0, a; I1).
1 { set(N, G) : group(G) } 1 :- vertex(N).