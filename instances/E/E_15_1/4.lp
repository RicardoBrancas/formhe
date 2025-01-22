%formhe-problem:E
%formhe-timestamp:1699278108
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix:1 { set(NODE, GROUP) : vertex(NODE), group(GROUP) } 1 :- vertex(NODE).  :- set(N1, G), set(N2, G), edge(N1, N2). group(a).

1 { set(NODE, GROUP) : vertex(NODE), group(GROUP) }  :- vertex(NODE).
 :- set(N1, N2), N2 == N1, edge(N1, N2).
group(b).
group(b).