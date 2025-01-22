%formhe-problem:E
%formhe-timestamp:1667846435
%formhe-selfeval-lines:0 2
%formhe-selfeval-fix: :- edge(V1, V2), set(V1, G), set(V2, G), V1 != V2. 1 { set(V, G) : g(G) } 1 :- vertex(V).

 :- edge(V1, V2), set(V1, 0), set(a, G), V1 != V2.
g(a); g(b).
 :- vertex(V).
1 { set(V, G) : g(G) } 1 :- vertex(V).