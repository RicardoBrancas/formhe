%formhe-problem:E
%formhe-timestamp:1667812516
%formhe-selfeval-lines:3
%formhe-selfeval-fix: :- edge(X, Y), not set(X, G), not set(Y, G), group(G).

1 { set(X, G) : group(G) } 1 :- vertex(X).
group(a).
group(b).
 :- edge(X, Y), group((X..G)), not set(Y, a), group(G).