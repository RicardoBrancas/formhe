%formhe-groundtruth:../../../mooshak/instances/1920_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShiLCAzKS4KZShiLCA0KS4KZShjLCAxKS4KZShjLCAzKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCAxKS4KZShhLCAyKS4KZShhLCAzKS4KZShiLCA0KS4KZShiLCA1KS4KZShiLCA2KS4KZShjLCA3KS4KZShjLCA4KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShhLCAzKS4KZShiLCA0KS4KZShiLCA1KS4KZShiLCA2KS4KZShjLCA3KS4KZShjLCA4KS4=

% % Mooshak-exercises in lab13 from ALC
% % Problem B


% generator
k { sel(set) } k :- e(X, element), not e(Y, element)

% tester
:- e(X, element), not e(Y, element)