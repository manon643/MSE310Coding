%% RAC - MBADMM - non negative least squares
clear all;close all;clc;
n = 10;
p = 20;
blocks = 5;
y = randn(n,1);
X = randn(n,p);
beta0 = pos(randn(p,1));
z0 = pos(randn(p,1));
mu0 = pos(randn(p,1));
eps = 1e-5;
k = 1;
err(k) = 10000;


while err(k) > eps

[beta_out,z_out,mu_out] = nnls(y,X,beta0,z0, mu0, blocks);
beta0 = beta_out;
z0 = z_out;
mu0 = mu_out;
beta(:,k) = beta_out;
z(:,k) = z_out;
mu(:,k) = mu_out;

k = k+1;
err(k) = norm(beta_out-z_out,2);
end

figure
plot(2:(k),err(2:k))

function [beta, z, mu] = nnls(y, X, beta, z, mu, blocks)
alpha = 1;
gamma = 1;
[n,p] = size(X); 
block_size = floor(p/blocks);
or = randperm(p);
% disp('BEFORE SHUFFLE')
% beta
% disp('after shuffle')
beta = beta(or);
z = z(or);

% for each block
    for j = 1:blocks
        idx_lb = (j-1)*block_size +1;
        idx_ub = idx_lb + block_size -1;
        indices = or(idx_lb:idx_ub);
        tmpX = X(:,indices);
%         disp('after reassign')
        beta(indices) = inv(1/n*tmpX'*tmpX + gamma*eye(block_size)) *(1/n* tmpX'*y - mu(indices) + gamma*z(indices));
    end
    
    %max(z,0) is what pos does
    z = pos(-mu./(gamma) + beta);
    mu = mu - alpha*(beta-z);
    
end