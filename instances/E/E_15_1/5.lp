%formhe-problem:E
%formhe-timestamp:1699278108
%formhe-selfeval-lines:1
%formhe-selfeval-fix:1 { set(NODE, GROUP) : vertex(NODE), group(GROUP) } 1 :- vertex(NODE).

 :- set(N1, G), set(N2, G), edge(N1, N2).
1 { edge(NODE, GROUP) : vertex(NODE), group(GROUP) } 1 :- vertex(NODE).
group(a).
group(b).