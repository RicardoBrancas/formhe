%formhe-problem:C
%formhe-timestamp:1698332122

element(X) :- e(X,S).
set(S) :- e(X,S).
:- sel(S), not set(S).
1{ sel(S) : e(X,S) } :- element(X).
k{ sel(S) : set(S) }k.
#show sel/1.