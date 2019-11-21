%
function [x,y]=RACADMMQP(Q,A,b,c,nb)
tic
toler=1.e-4;
[m,n]=size(A);
beta=1;
x=ones(n,1);
s=x;
y=zeros(m,1);
z=zeros(n,1);
nv=floor(n/nb);
% greate two vectos: the gradient of objective and constraint LHS
Qxc=Q*x+c;
Axb=A*x-b;
%
iter=0;
for k=1:300,
  % randomly reorder all variables
  or=randperm(n);
% Update x following a random order
  for ii=1:nb,
      il=(ii-1)*nv+1;
      iu=il+nv-1;
      % find the nv indexes of the (ii)th block according to or
      p=or(il:iu);
      % substract them from the two vectors
      Qxc=Qxc-Q(:,p)*x(p);
      Axb=Axb-A(:,p)*x(p);
      % construct sub-A matrix according to p
      Ap=A(:,p);
      % Construct the gradient vector of x(p)
      cc=Qxc(p)-z(p)-beta*s(p)-Ap'*(y-beta*Axb);
      % update x(p)
      x(p)=-(Q(p,p)+beta*(Ap'*Ap)+beta*eye(nv))\cc;
      % add them back to the two vectors
      Qxc=Qxc+Q(:,p)*x(p);
      Axb=Axb+A(:,p)*x(p);
      % go to the next block
  end
% Update s
  cc=x-(1/beta)*z;
  s=max(0,cc);
% Update the two multipliers
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
% Random Sample-Without-Replacement Cyclic Multi-Block ADMM for solving 
% convex quadratic program:
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