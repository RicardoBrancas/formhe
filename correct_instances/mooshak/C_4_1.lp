%formhe-groundtruth:../../mooshak/instances/1819_2.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSg0LCBjKS4KZSg1LCBjKS4=

% % Mooshak-exercises in lab13 from ALC
% % Problem C
% % constants
% #const k = 2 .

% % facts
% e(1, a). e(2, a). e(3, a).

% e(2, b). e(4, b).

% e(4, c). e(5, c).

% generator
k { sel(SET) : e(_, SET) } k .

% tester
:- sel(SET1), sel(SET2), e(ELEMENT, SET1), e(ELEMENT, SET2), SET1 != SET2 .