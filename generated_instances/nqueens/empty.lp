%formhe-groundtruth:../../examples/nqueens.lp
% generate
#const n = 8.

row(1..n).
col(1..n).

n { queen(I, J) : row(I), col(J) } n.


% test

%:- queen(I1, J1), queen(I2, J2), (I1, J1) != (I2, J2), I1 == I2.
%:- queen(I1, J1), queen(I2, J2), (I1, J1) != (I2, J2), J1 == J2.
%:- queen(I1, J1), queen(I2, J2), (I1, J1) != (I2, J2), I1+J1 == I2+J2.
%:- queen(I1, J1), queen(I2, J2), (I1, J1) != (I2, J2), I1-J1 == I2-J2.