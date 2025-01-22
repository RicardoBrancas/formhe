%formhe-problem:E
%formhe-timestamp:1699277272
%formhe-selfeval-lines:2
%formhe-selfeval-fix:1{set(NODE, GROUP): group(GROUP)}1 :- vertex(NODE).

group(a).
group(b).
1{set(NODE, GROUP): vertex(NODE), group(GROUP)}1.
:-set(N1,G), set(N2,G), edge(N1,N2).