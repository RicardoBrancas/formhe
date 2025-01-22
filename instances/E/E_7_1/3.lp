%formhe-problem:E
%formhe-timestamp:1667812516
%formhe-selfeval-lines:0 1 3
%formhe-selfeval-fix: :- edge(X, Y), not set(X, G), not set(Y, G), group(G). group(a). 1 { set(X, G) : group(G) } 1 :- vertex(X).

 :- edge(X, Y), not edge(X, G), -set(a, G), group(G), vertex((b..G)).
set(a, 0).
group(b).
 :- vertex(X).