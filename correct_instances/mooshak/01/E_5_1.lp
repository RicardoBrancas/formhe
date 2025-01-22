%formhe-problem:E
%formhe-timestamp:1667931098

%set como definir este set
%2{s(A): vertex(A)}2.
s(a). s(b).

1{set(A,B): s(B) }1:- vertex(A).

:- vertex(A), s(B), vertex(C), s(D), set(A,B), set(C,D), edge(A,C), B==D.

#show set/2.