%formhe-problem:E
%formhe-timestamp:1698332121
%formhe-selfeval-lines:0
%formhe-selfeval-fix: :- edge(U, V), set(U, b), set(V, b).

 :- edge(U, V), set(U, b), edge(V, a).
1 { set(U, a); set(U, b) } 1 :- vertex(U).
 :- edge(U, V), set(U, a), set(V, a).