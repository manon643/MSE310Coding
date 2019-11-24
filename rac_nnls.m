
function [beta, z, mu] = rac_nnls(y, X, beta, z, mu, blocks,gamma)
[n,p] = size(X); 
block_size = floor(p/blocks);
or = randperm(2*p);

% for each block
    for j = 1:blocks
        idx_lb = (j-1)*block_size +1;
        idx_ub = idx_lb + block_size -1;
        indices = or(idx_lb:idx_ub);
        for ii = 1:block_size
            val = indices(ii);
            
            if val < p %update beta
                tmpX = X(:,indices(ii));
                tmp(ii) = (1/n*tmpX'*tmpX + gamma) \ (1/n* tmpX'*y + mu(indices(ii)) + gamma*z(indices(ii)));
            else %update z
                if val == 2*p
                    val_idx = val-p;
                elseif val == p
                    val_idx = val-p+1;
                else
                    val_idx = val-p+1;
                end
                tmp(ii) = pos(-mu(val_idx) / gamma + beta(val_idx));
            end
        end
        for jj = 1:block_size
            if indices(jj) < p
                beta(indices(jj)) = tmp(jj);
            else
                if indices(jj) == 2*p
                    val_idx = indices(jj)-p;
                elseif indices(jj) == p
                    val_idx = indices(jj)-p+1;
                else
                    val_idx = indices(jj)-p+1;
                end
                z(val_idx) = tmp(jj);
            end
        end
        
%         tmpX = X(:,indices);
%         beta(indices) = -inv(1/n*tmpX'*tmpX + gamma*eye(block_size)) *(1/n* tmpX'*y - mu(indices) - gamma*z(indices));
%         beta(indices) = (1/n*tmpX'*tmpX + gamma*eye(block_size)) \ (1/n* tmpX'*y + mu(indices) + gamma*z(indices));
        
    end
    %max(z,0) is what pos does
%     z = pos(-mu./(gamma) + beta);
    mu = mu - gamma*(beta-z);
    
end
