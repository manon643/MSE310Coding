%% RAC - MBADMM - non negative least squares
% MSE 310 Linear Programming
% project 3
% Stephen Palmieri


%% RAC
clear all;close all;clc;
n = 10;
p = 20;
% n = 1000;
% p = 2000;
% n = 100;
% p = 200;
blocks = 5;
rng(5)
y = sprandn(n,1,.1);
X = sprandn(n,p,.1);
beta_true = pos(sprandn(p,1,.1));
y = X*beta_true;
beta01 = pos(sprandn(p,1,.1));
z01 = pos(sprandn(p,1,.1));
mu01 = pos(sprandn(p,1,.1));
k = 1;
err(k) = norm(beta_true-beta01,2);
err_bz(k) = norm(beta01-z01,2);
toler = 1e-4;
maxIter = 1000;
gammas = [.01, .1, 1, 10];

for jj = 1:length(gammas)
    beta0 = beta01;z0 = z01;mu0 = mu01;
    tic
    for ii = 1:maxIter
        
        [beta_out,z_out,mu_out] = rac_nnls(y,X,beta0,z0, mu0, blocks,gammas(jj));
        beta0 = beta_out; z0 = z_out;mu0 = mu_out;        
        obj(k,jj) = 1/(2*n) * (y-X*beta_out)'*(y-X*beta_out);
        obj_al(k,jj) = obj(k,jj) + gammas(jj)/2*norm(beta_out-z_out,2)^2 - mu_out'*(beta_out-z_out);
        k = k+1;
        err(k,jj) = norm(beta_true-beta_out,2);
        err_bz(k,jj) = norm(beta_out-z_out,2);
        % if abs(err(k)-err(k-1))  < toler
        if abs(err(k,jj)) < toler
            disp('below tolerance')
            break
        else
        end
        
    end
    t(jj) = toc;
    k = 1;
end
%verifying non-negativity constraints
% beta_out >= 0
% z_out >= 0
figure
plot(gammas,t, 'b*')
xlabel('\gamma values')
ylabel('time (s)')
title('Time elapsed for RAC')
mean(t)

%% plotting RAC

figure
for iii = 1:length(gammas)
    plot(err(2:200,iii))
    hold on
    
end
xlabel('iterations')
ylabel('error')
title('l-2 norm \beta error from true value')
legend('.01', '.1','1','10')
hold off

figure
for iii = 1:length(gammas)
    plot(err_bz(2:200,iii))
    hold on
    
end
xlabel('iterations')
ylabel('error')
title('l-2 norm (\beta - z) error')
legend('.01', '.1','1','10')
hold off

figure
for iii = 1:length(gammas)
    plot(obj(2:200,iii))
    hold on
    %     plot(obj_al)
    %     legend('original obj','augmented Lagrangian')
    
end
xlabel('iterations')
ylabel('objective loss')
title('Non-negative Least Squares Objective Loss vs Iterations using RAC-MBADMM')
legend('.01', '.1','1','10')
hold off

%% Randomly Permute Comparison - Section IV
clear all;close all;clc;
n = 100;
p = 200;
% n = 1000;
% p = 2000;
% n = 10;
% p = 20;
blocks = p;

rng(5)
y = sprandn(n,1,.1);
X = sprandn(n,p,.1);

beta_true = pos(sprandn(p,1,.1));
y = X*beta_true;

beta01 = pos(sprandn(p,1,.1));
z01 = pos(sprandn(p,1,.1));
mu01 = pos(sprandn(p,1,.1));
k = 1;
err(k) = norm(beta_true-beta01,2);
err_bz(k) = norm(beta01-z01,2);
toler = 1e-3;
maxIter = 1000;
gammas = [.01, .1, 1, 10];

for jj = 1:length(gammas)
    beta0 = beta01;z0 = z01;mu0 = mu01;
    tic;
    for ii = 1:maxIter
        
        [beta_out,z_out,mu_out] = rp_nnls(y,X,beta0,z0, mu0, blocks,gammas(jj));
        beta0 = beta_out; z0 = z_out;mu0 = mu_out;        
        obj(k,jj) = 1/(2*n) * (y-X*beta_out)'*(y-X*beta_out);
        obj_al(k,jj) = obj(k,jj) + gammas(jj)/2*norm(beta_out-z_out,2)^2 - mu_out'*(beta_out-z_out);
        k = k+1;
        err(k,jj) = norm(beta_true-beta_out,2);
        err_bz(k,jj) = norm(beta_out-z_out,2);
        % if abs(err(k)-err(k-1))  < toler
        if abs(err(k,jj)) < toler
            disp('below tolerance')
            break
        else
        end
        
    end
    t(jj) = toc;
    k = 1;
end
%verifying non-negativity constraints
% beta_out >= 0
% z_out >= 0
figure
plot(gammas,t, 'b*')
xlabel('\gamma values')
ylabel('time (s)')
title('Time elapsed for RP')
mean(t)
%% plotting RP
figure
for iii = 1:length(gammas)
    plot(err(2:200,iii))
    hold on
    
end
xlabel('iterations')
ylabel('error')
title('l-2 norm \beta error from true value')
legend('.01', '.1','1','10')
hold off

figure
for iii = 1:length(gammas)
    plot(err_bz(2:200,iii))
    hold on
    
end
xlabel('iterations')
ylabel('error')
title('l-2 norm (\beta - z) error')
legend('.01', '.1','1','10')
hold off

figure
for iii = 1:length(gammas)
    plot(obj(2:200,iii))
    hold on
    %     plot(obj_al)
    %     legend('original obj','augmented Lagrangian')
    
end
xlabel('iterations')
ylabel('objective loss')
title('Non-negative Least Squares Objective Loss vs Iterations using RP-MBADMM')
legend('.01', '.1','1','10')
hold off
