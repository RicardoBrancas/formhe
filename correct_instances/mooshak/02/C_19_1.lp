%formhe-problem:C
%formhe-timestamp:1698333409

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