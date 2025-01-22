%formhe-problem:E
%formhe-timestamp:1667812516
%formhe-selfeval-lines:3 4
%formhe-selfeval-fix: :- edge(X, Y), not set(X, G), not set(Y, G), group(G).

group(a).
group(b).
1 { set(X, G) : group(G) } 1 :- vertex(X).
 :- edge(X, Y), not set(X, G), -set(Y, G), group(G), set(0; 1, 0; 0).
set(1, 1).