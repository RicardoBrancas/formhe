%formhe-problem:B
%formhe-timestamp:1667774703

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