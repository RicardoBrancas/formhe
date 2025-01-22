%formhe-problem:A
%formhe-timestamp:1667769852
%formhe-selfeval-lines:0 1
%formhe-selfeval-fix:k(1..k).

% % Mooshak-exercises in lab13 from ALC
% % Problem A
#const n = 3.


k(1..n).

1 { assign(N, C) : k(C) } 1 :- e(_, N).

1 { assign(N, C) : k(C) } 1 :- e(N, _).


:- assign(N, C), assign(A, C), e(N,A).