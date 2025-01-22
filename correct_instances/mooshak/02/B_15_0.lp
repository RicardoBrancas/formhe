%formhe-problem:B
%formhe-timestamp:1699275400

set(S):- e(S, _).
k {sel(S): set(S)} k.
:-sel(S1), sel(S2), e(S1,E), e(S2,E), S1 != S2.