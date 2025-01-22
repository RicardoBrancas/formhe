%formhe-problem:E
%formhe-timestamp:1699278108
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix:1 { set(NODE, GROUP) : vertex(NODE), group(GROUP) } 1 :- vertex(NODE).  :- set(N1, G), set(N2, G), edge(N1, N2).

group(a).
group(b).
1 { set(NODE, GROUP) : vertex(_), group(GROUP) } 1 :- vertex(NODE).
 :- set(N1, _), set(G, GROUP), edge(N1, _).