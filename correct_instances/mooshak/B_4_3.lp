%formhe-groundtruth:../../mooshak/instances/1920_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShiLCAzKS4KZShiLCA0KS4KZShjLCAxKS4KZShjLCAzKS4K

% % Mooshak-exercises in lab13 from ALC
% % Problem B
% % constants
% #const k = 2.

% % facts
% % these are the sets a={1, 2}, b={3, 4}, c={1, 3}
% e(a, 1).
% e(a, 2).

% e(b, 3).
% e(b, 4).

% e(c, 1).
% e(c, 3).

% generator
k {sel(SET): e(SET, _)} k .

% tester
:- sel(SET1), sel(SET2), e(SET1, V1), e(SET2, V2), V1 == V2, SET1 != SET2 .