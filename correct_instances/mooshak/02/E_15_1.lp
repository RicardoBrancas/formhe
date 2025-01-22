%formhe-problem:E
%formhe-timestamp:1699278109

group(a).
group(b).
1{set(NODE, GROUP): vertex(NODE), group(GROUP)}1:-vertex(NODE).
:-set(N1,G), set(N2,G), edge(N1,N2).