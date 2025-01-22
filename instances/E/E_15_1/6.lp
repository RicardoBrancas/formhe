%formhe-problem:E
%formhe-timestamp:1699278108
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix:group(a).  :- set(N1, G), set(N2, G), edge(N1, N2).

1 { set(NODE, GROUP) : vertex(NODE), group(GROUP) } 1 :- vertex(NODE).
group(b).
group(1).
 :- set(a, G), edge(N1, G), edge(0, N2).