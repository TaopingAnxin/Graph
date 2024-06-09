function [Precision, Recall, JS]=pre_rec(A,B)
[l1, l2] = size(A);
A=A(:);
B=B(:);
C=A(:)+B(:);
TP=sum(B(A==1)==1); % AB %true positive
FN=sum(B==1)-TP; % true negative
FP=sum(A==1)-TP; % false positvie
TT=l1*l2-sum(C==0);
Precision=TP/(TP+FP);
Recall=TP/(TP+FN);
JS=TP/TT;
return