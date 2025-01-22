%formhe-problem:D
%formhe-timestamp:1699276220
%formhe-selfeval-lines:2
%formhe-selfeval-fix:1{sel(N):nodes(N)}k.

nodes(X):-e(X,_).
nodes(Y):-e(_,Y).
k{sel(N):nodes(N)}k.
:-e(X,Y), not sel(X), not sel(Y).