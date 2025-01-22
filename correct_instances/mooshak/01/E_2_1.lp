%formhe-problem:E
%formhe-timestamp:1667941046

group(a).
group(b).

:- edge(X, Y), set(X, Z), set(Y, Z), vertex(X), vertex(Y), group(Z), X!= Y.
:- set(X, Z), set(X, W), Z!= W, group(Z), group(W).

1{set(X, Z): vertex(X), group(Z)}1 :- vertex(X).

#show set/2.