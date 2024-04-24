%formhe-groundtruth:../../../mooshak/instances/1819_2.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSg0LCBjKS4KZSg1LCBjKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSgzLCBjKS4KZSg0LCBjKS4KZSg0LCBkKS4KZSg1LCBkKS4=
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix:contains(SET1, SET2, V) :- e(V, SET1), e(V, SET2), SET1 != SET2. not sel(SET1) :- sel(SET2), contains(SET1, SET2, V), SET1 != SET2.

k { sel(SET) : e(_, SET) } k.
contains(SET1, V, V) :- e(V, SET1), e(V, SET2), SET1 != SET2.
-sel(SET1) :- sel(SET2), contains(SET1, SET2, V), SET1 != SET2.