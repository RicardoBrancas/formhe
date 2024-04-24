%formhe-groundtruth:../01/B_0_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShiLCAzKS4KZShiLCA0KS4KZShjLCAxKS4KZShjLCAzKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCAxKS4KZShhLCAyKS4KZShhLCAzKS4KZShiLCA0KS4KZShiLCA1KS4KZShiLCA2KS4KZShjLCA3KS4KZShjLCA4KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShhLCAzKS4KZShiLCA0KS4KZShiLCA1KS4KZShiLCA2KS4KZShjLCA3KS4KZShjLCA4KS4=

% Default
%% #const k = 2.

% Sets
%% e(a, 1). e(a, 2).
%% e(b, 3). e(b, 4).
%% e(c, 1). e(c, 3).

% Generate
{ sel(X) : e(X,_) } = k.

% Test
:- sel(A), sel(B), A != B,
   e(A,X), e(B,X).

% Display
#show sel/1.