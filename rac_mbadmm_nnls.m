%% RAC - MBADMM - non negative least squares
% MSE 310 Linear Programming
% project 3
% Stephen Palmieri


%% RAC
clear all;close all;clc;
% n = 10;
% p = 20;
n = 100;
p = 200;
blocks = 5;
rng(5)
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
toler = 1e-4;
maxIter = 1000;
gammas = [.1, 1, 10,1000];

for jj = 1:length(gammas)
    for ii = 1:maxIter
        
        [beta_out,z_out,mu_out] = rac_nnls(y,X,beta0,z0, mu0, blocks,gammas(jj));
        beta0 = beta_out; z0 = z_out;mu0 = mu_out;
        %         beta(:,k) = beta_out;z(:,k) = z_out; mu(:,k) = mu_out;
        
        obj(k,jj) = 1/(2*n) * (y-X*beta_out)'*(y-X*beta_out);
        obj_al(k,jj) = obj(k,jj) + gammas(jj)/2*norm(beta_out-z_out,2)^2 - mu_out'*(beta_out-z_out);
        k = k+1;
        
        % err(k) = norm(beta_out-z_out,2);
        err(k,jj) = norm(beta_true-beta_out,2);
        err_bz(k,jj) = norm(beta_out-z_out,2);
        % if abs(err(k)-err(k-1))  < toler
        if err(k,jj) < toler
            disp('below tolerance')
            break
        else
        end
        
    end
    k = 1;
end
%verifying non-negativity constraints
beta_out > 0
z_out > 0

figure
for iii = 1:length(gammas)
    plot(err(:,iii))
    hold on
    
end
xlabel('iterations')
ylabel('error')
title('l-2 norm \beta error from true value')
legend('.1','1','10','1000')
hold off

figure
for iii = 1:length(gammas)
    plot(err_bz(:,iii))
    hold on
    
end
xlabel('iterations')
ylabel('error')
title('l-2 norm (\beta - z) error')
legend('.1','1','10','1000')
hold off

figure
for iii = 1:length(gammas)
    plot(obj(:,iii))
    hold on
    %     plot(obj_al)
    %     legend('original obj','augmented Lagrangian')
    
end
xlabel('iterations')
ylabel('objective loss')
title('Non-negative Least Squares Objective Loss vs Iterations using RAC-MBADMM')
legend('.1','1','10','1000')
hold off

%% Randomly Permute Comparison - Section IV
clear all;close all;clc;
n = 100;
p = 200;
blocks = p;

rng(5)
y = sprandn(n,1,.1);
X = sprandn(n,p,.1);

beta_true = pos(sprandn(p,1,.1));
y = X*beta_true;

beta0 = pos(sprandn(p,1,.1));
z0 = pos(sprandn(p,1,.1));
mu0 = pos(sprandn(p,1,.1));
gammas = [.1, 1, 10,1000];

k = 1;
err(k) = norm(beta_true-beta0,2);
err_bz(k) = norm(beta0-z0,2);
toler = 1e-3;
maxIter = 1000;

