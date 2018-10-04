function [K1,K2]=sptest(N)
% create a NxN random matrix and recast matlab sparse matrix into
% dict/struc
%  K1 is real and K2 is complex
R1 = sprand(N,N,0.5);
R2 = R1 *(1+ 0.1i);
[K1,K2]=Recast4py(R1,R2);