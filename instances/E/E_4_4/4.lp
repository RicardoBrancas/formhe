%formhe-problem:E
%formhe-timestamp:1667846435
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix:1 { set(V, G) : g(G) } 1 :- vertex(V). 1 { set(V, G) : g(G) } 1 :- vertex(V).

 :- edge(V1, V2), set(V1, G), set(V2, G), V1 != V2.
g(a); g(b).
1 { set(V, 0) : g(G) } 1 :- vertex(V), edge(I0; 0, (b..0)).
 :- vertex(a).