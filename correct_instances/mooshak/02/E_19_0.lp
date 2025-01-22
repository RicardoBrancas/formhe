%formhe-problem:E
%formhe-timestamp:1698344187

%% vertex(a). vertex(b). vertex(c). vertex(d). vertex(e).
%% edge(a,b). edge(b,c). edge(c,d). edge(d,a). edge(d,e).

group(a). group(b).

{ set(N, G) : group(G) } = 1 :- vertex(N).

:- set(X, A), set(Y, A), edge(X, Y).

#show set/2.