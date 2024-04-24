%formhe-groundtruth:../01/C_0_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSg0LCBjKS4KZSg1LCBjKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSgzLCBjKS4KZSg0LCBjKS4KZSg0LCBkKS4KZSg1LCBkKS4=

%% #const k = 2.

%% e(1, a). e(2, a). e(3, a).
%% e(2, b). e(4, b).
%% e(4, c). e(5, c).

{ sel(S) : e(_, S) } = k.

%% first approach
%% :- e(E, _), { sel(S) : e(E, S)} 0.

%% second approach
selected(X) :- sel(S), e(X, S).
:- e(X, S), not selected(X).


#show sel/1.