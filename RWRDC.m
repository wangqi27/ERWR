%MATLAB code
%input:adjacency matrix A
%output:efficacy score S
function S=RWRDC(A)
c=0.05;%restart probability
[cc,cc]=size(A);
w=inv(diag(sum(A,2)))*A;
for k=1:cc
p0=zeros(cc,1);
p0(k)=1; 
p1=p0;
p(:,k)=(1-c)*w'*p0+c*p0;
while max(abs(p(:,k)-p1))>10^(-6)
    p1=p(:,k);
    p(:,k)=(1-c)*w'*p1+c*p0;
end
end
S=p+p';
end
