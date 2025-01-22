%formhe-problem:E
%formhe-timestamp:1668079604
%formhe-selfeval-lines:2
%formhe-selfeval-fix:1 { set(NODE, GROUP) : group(GROUP) } 1 :- vertex(NODE).

group(a).
group(b).

set(NODE, GROUP) :- vertex(NODE), group(GROUP).

:- set(N, G), set(N1, G), N1 != N, edge(N, N1).