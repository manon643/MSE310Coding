%
function [x,y]=RPADMMQP(Q,A,b,c,nb)
tic
toler=1.e-4;
[m,n]=size(A);
beta=1;
x=ones(n,1);
s=x;
y=zeros(m,1);
z=zeros(n,1);
nv=floor(n/nb);
ATA=zeros(nv,n);
% store block-wise inverse
for ii=1:nb,
    il=(ii-1)*nv+1;
    iu=il+nv-1;
    AB=A(:,il:iu);
    ATA(:,il:iu)=inv(Q(il:iu,il:iu)+beta*(AB'*AB)+beta*eye(nv));
end;
clear AB;
% greate two vectos: the gradient of objective and constraint LHS
Qxc=Q*x+c;
Axb=A*x-b;
%
iter=0;
for k=1:300,
  or=randperm(nb);
% Update x following a random order
  for ii=1:nb,
      il=(or(ii)-1)*nv+1;
      iu=il+nv-1;
      Qxc=Qxc-Q(:,il:iu)*x(il:iu);
      Axb=Axb-A(:,il:iu)*x(il:iu);
      cc=Qxc(il:iu)-z(il:iu)-beta*s(il:iu)-A(:,il:iu)'*(y-beta*Axb);
      x(il:iu)=-ATA(:,il:iu)*cc;
      Qxc=Qxc+Q(:,il:iu)*x(il:iu);
      Axb=Axb+A(:,il:iu)*x(il:iu);
  end
% Update s
  cc=x-(1/beta)*z;
  s=max(0,cc);
% Update multipliers
  y=y-beta*Axb;
  z=z-beta*(x-s);
%
  iter=iter+1;
  % Check stopping criteria
  if (norm(Axb)+norm(x-s))/(1+norm(x))<toler, break, end;
end;
toc
x=s;
iter=iter
% Randomly Permuted Multi-Block ADMM for solving convex quadratic program
%
%      minimize      0.5x'Qx + c'x
%      subject to     Ax     = b,  (y dimension m)
%                     x -  s = 0,  (z dimesnion n)
%                     s>=0
%
%      Input: A, Q, b, c, nb(number of blocks such that number of variables
%                            divided by nb is an integer)
%             
%      Output:primal x, dual y
%          