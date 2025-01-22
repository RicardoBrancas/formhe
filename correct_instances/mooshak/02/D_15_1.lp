%formhe-problem:D
%formhe-timestamp:1699276602

nodes(X):-e(X,_).
nodes(Y):-e(_,Y).
1{sel(N):nodes(N)}k.
:-e(X,Y), not sel(X), not sel(Y).