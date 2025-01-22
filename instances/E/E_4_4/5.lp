%formhe-problem:E
%formhe-timestamp:1667846435
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix: :- edge(V1, V2), set(V1, G), set(V2, G), V1 != V2. 1 { set(V, G) : g(G) } 1 :- vertex(V).

g(a); g(b).
 :- set(V1, V2), set(V1, G), set(1, _), set(V1, V2).
1 { set(G, G) : g(G) } 1 :- vertex(V).
1 { set(V, G) : g(G) } 1 :- vertex(V).