%formhe-problem:E
%formhe-timestamp:1698332121
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- edge(U, V), set(U, b), set(V, b).

 :- edge(U, V), set(U, a), set(V, a).
 :- edge(U, V), set(b, b), set(V, b).
1 { set(U, a); set(U, b) } 1 :- vertex(U).