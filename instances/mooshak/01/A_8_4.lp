%formhe-problem:A
%formhe-timestamp:1667511133
%formhe-selfeval-lines:0
%formhe-selfeval-fix:k(1..k).

k(1..n).

1 { assign(N, C) : k(C) } 1 :- e(_, N).

1 { assign(N, C) : k(C) } 1 :- e(N, _).


:- assign(N, C), assign(A, C), e(N,A).