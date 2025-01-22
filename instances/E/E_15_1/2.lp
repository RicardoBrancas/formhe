%formhe-problem:E
%formhe-timestamp:1699278108
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix: :- set(N1, G), set(N2, G), edge(N1, N2). 1 { set(NODE, GROUP) : vertex(NODE), group(GROUP) } 1 :- vertex(NODE). group(a).

 :- set(N1, G), edge(N2, G), edge(N1, N2), set(b, 1; I1).
1 { set(NODE, GROUP) : vertex(b), group(GROUP) } 1 :- vertex(NODE).
set(a, (1..0)).
group(b).