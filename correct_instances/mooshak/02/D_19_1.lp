%formhe-problem:D
%formhe-timestamp:1698344187

%% #const k = 2.

%% e(1,2). e(1,3). e(4,3). e(4,5).

v(X) :- e(X,_).
v(X) :- e(_,X).

{ sel(X) : v(X) } k.
:- e(X,Y), not sel(X), not sel(Y).

#show sel/1.