%formhe-problem:C
%formhe-timestamp:1698332765

%% #const k = 2.

%% e(1, a). e(2, a). e(3, a).
%% e(2, b). e(4, b).
%% e(4, c). e(5, c).

{ sel(S) : e(_, S) } = k.

:- e(E, _), { sel(S) : e(E, S)} 0.
%% e(E, _) :- 1 { sel(S): e(E, S) }.

#show sel/1.