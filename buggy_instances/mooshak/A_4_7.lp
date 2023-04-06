%formhe-groundtruth:../../mooshak/instances/1920_2.lp
%formhe-domain-predicates:assign/2
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShhLCBkKS4KZShjLCBkKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShiLCBjKS4KZShjLCBkKS4KZShjLCBlKS4KZShjLCBmKS4KZShjLCBnKS4=
%formhe-selfeval-lines:2
%formhe-selfeval-fix:c(1..k). v(N) :- e(_, N). v(N) :- e(N, _). assign(X, I) :- color(X, I).

% % Mooshak-exercises in lab13 from ALC
% % Problem A

% % facts
% col(1..k).

% % generate
% { assign(X,C) : col(C) } == 1 :- node(X).

% % test
% :- edge(X,Y), assign(X,C), assign(Y,C).

c(1..n).
1 {color(X,I) : c(I)} 1 :- v(X).             
:- color(X,I), color(Y,I), e(X,Y), c(I).

v(1..n). % 1,...,100 are vertices
% e(1,55). % there is an edge from 1 to 55