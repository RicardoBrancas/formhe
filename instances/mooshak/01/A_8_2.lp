%formhe-problem:A
%formhe-timestamp:1667510350
%formhe-selfeval-lines:
%formhe-selfeval-fix:k(1..k).

1 { assign(N, C) : k(C) } 1 :- e(_, N).

1 { assign(N, C) : k(C) } 1 :- e(N, _).


:- assign(N, C), assign(A, C), C==C1, e(N,A).