%formhe-problem:B
%formhe-timestamp:1698332302

% Default
%% #const k = 2.

% Sets
%% e(a, 1). e(a, 2).
%% e(b, 3). e(b, 4).
%% e(c, 1). e(c, 3).

% Generate
{ sel(X) : e(X,_) } = k.

% Test
:- sel(A), sel(B), A != B, e(A,X), e(B,X).

% Display
#show sel/1.