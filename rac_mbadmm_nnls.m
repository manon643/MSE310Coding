%% RAC - MBADMM - non negative least squares
clear all;close all;clc;
n = 100;
p = 2000;
blocks = 10;
y = sprandn(n,1,.1);
X = sprandn(n,p,.1);

beta_true = pos(sprandn(p,1,.1));
y = X*beta_true;

beta0 = pos(sprandn(p,1,.1));
z0 = pos(sprandn(p,1,.1));
mu0 = pos(sprandn(p,1,.1));

k = 1;
err(k) = norm(beta_true-beta0,2);
err_bz(k) = norm(beta0-z0,2);
toler = 1e-3;
maxIter = 1000;
for ii = 1:maxIter

[beta_out,z_out,mu_out] = nnls(y,X,beta0,z0, mu0, blocks);
beta0 = beta_out;
z0 = z_out;
mu0 = mu_out;
beta(:,k) = beta_out;
z(:,k) = z_out;
mu(:,k) = mu_out;
obj(k) = 1/(2*n) * (y-X*beta_out)'*(y-X*beta_out);
k = k+1;

% err(k) = norm(beta_out-z_out,2);
err(k) = norm(beta_true-beta_out,2);
err_bz(k) = norm(beta_out-z_out,2);
if err(k) <toler
    disp('below tolerance')
    break
else
end

end

figure
plot(1:k,err)
xlabel('iterations')
ylabel('error')
title('l-2 norm \beta error')

figure
plot(1:k,err_bz)
xlabel('iterations')
ylabel('error')
title('l-2 norm (\beta - z) error')

figure
plot(obj)
xlabel('iterations')
ylabel('objective loss')
title('Non-negative Least Squares Objective Loss vs Iterations using RAC-MBADMM')

function [beta, z, mu] = nnls(y, X, beta, z, mu, blocks)
% alpha = 1.8; %this should be the same as the gamma according to their
% paper

gamma = 10;
% gamma = 1000;
[n,p] = size(X); 
block_size = floor(p/blocks);
or = randperm(p);

beta = beta(or);
z = z(or);

% for each block
    for j = 1:blocks
        idx_lb = (j-1)*block_size +1;
        idx_ub = idx_lb + block_size -1;
        indices = or(idx_lb:idx_ub);
        tmpX = X(:,indices);
%         beta(indices) = -inv(1/n*tmpX'*tmpX + gamma*eye(block_size)) *(1/n* tmpX'*y - mu(indices) - gamma*z(indices));
        beta(indices) = (1/n*tmpX'*tmpX + gamma*eye(block_size)) \ (1/n* tmpX'*y + mu(indices) + gamma*z(indices));
    end
    
    %max(z,0) is what pos does
    z = pos(-mu./(gamma) + beta);
    mu = mu - gamma*(beta-z);
    
end