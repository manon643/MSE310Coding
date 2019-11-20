%% RAC - MBADMM - non negative least squares
clear all;close all;clc;
n = 10;
p = 20;
blocks = 5;
y = randn(n,1);
X = randn(n,p);
beta0 = randn(p,1);
z0 = pos(randn(p,1));
mu0 = pos(randn(p,1));
eps = 1e-5;
k = 1;
err(k) = 10000;
% 
% % while err(k) > eps
% % 
[beta_out,z_out,mu_out] = nnls(y,X,beta0,z0, mu0, blocks);
% % beta0 = beta_out;
% % z0 = z_out;
% % mu0 = mu_out;
% % beta(:,k) = beta_out;
% % z(:,k) = z_out;
% % mu(:,k) = mu_out;
% % 
% % k = k+1;
% % err(k) = norm(beta_out-z_out,2);
% % end
% % 
% % figure
% % plot(2:(k),err(2:k))

function [beta, z, mu] = nnls(y, X, beta, z, mu, blocks)
alpha = 1.2;
gamma = .3;
[n,p] = size(X); 
block_size = p/blocks;
% beta
block_shuffle = randsample(1:blocks,blocks);
permutation = randperm(p);
beta = beta(permutation);
z = z(permutation);
beta = reshape(beta,[block_size,blocks]);
beta = beta(:,block_shuffle);


% for each block
    for j = block_shuffle
%         idx = shuffle(j*num_blocks:j*num_blocks+4);
        idx = fliplr(j*block_size:-1:j*block_size-(block_size-1));
        tmpX = X(:,idx);
        beta(:,j) = inv(2*tmpX'*tmpX + gamma*eye(block_size)) *(2*tmpX'*y - mu(idx) + 2*gamma*z(idx));
    end
    
    
    beta = reshape(beta,[p,1]);
    z = pos(mu./(2*gamma) + beta);
    mu = mu - alpha*(beta-z);
    
end