%formhe-problem:E
%formhe-timestamp:1667846435
%formhe-selfeval-lines:0
%formhe-selfeval-fix:g(a); g(b).

g(a); vertex(b).
 :- edge(V1, V2), set(V1, G), set(V2, G), V1 != V2.
1 { set(V, G) : g(G) } 1 :- vertex(V).
1 { set(V, G) : g(G) } 1 :- vertex(V).