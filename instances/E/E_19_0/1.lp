%formhe-problem:E
%formhe-timestamp:1698344186
%formhe-selfeval-lines:0 2 3
%formhe-selfeval-fix:group(a). 1 { set(N, G) : group(G) } 1 :- vertex(N).  :- set(X, A), set(Y, A), edge(X, Y).

vertex(a).
group(b).
1 { set(N, G) : group(G) } 1 :- vertex(N), -set(Y, N).
 :- set(X, A), set(Y, A), edge(Y, Y), 1 == 0.