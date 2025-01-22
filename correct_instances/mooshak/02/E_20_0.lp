%formhe-problem:E
%formhe-timestamp:1698332122

:- edge(U,V), set(U,a), set(V,a).
:- edge(U,V), set(U,b), set(V,b).
1{ set(U,a);set(U,b) }1 :- vertex(U).
#show set/2.