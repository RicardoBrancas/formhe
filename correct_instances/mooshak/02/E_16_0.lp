%formhe-problem:E
%formhe-timestamp:1698335931

edge(Y, X) :- edge(X, Y).

1 { set(X, a) ; set(X, b) } 1 :- vertex(X).

:- edge(X, Y), set(X, S), set(Y, S).

#show set/2.