%formhe-problem:E
%formhe-timestamp:1699278108
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- set(N1, G), set(N2, G), edge(N1, N2).

1 { set(NODE, GROUP) : vertex(NODE), group(GROUP) } 1 :- vertex(NODE).
 :- set(N1, N2), set(N2, GROUP), edge(N1, N2).
group(a).
group(b).