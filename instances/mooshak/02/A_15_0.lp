%formhe-problem:A
%formhe-timestamp:1698327323

assign(N1, color):- e(N2,N1), not assign(N2,color)
assign(N1, color):- e(N1,N2), not assign(N2,color)
1{assign(N, 1..k)}1:-node(N)