function [R,B,Q] = FAAN(X,r,thres,init)
%%%%%%      Date: 24/01/2023
%%%%        Code written by Prabhu Babu, email:prabhubabu@care.iitd.ac.in
%%%%%%%%%%  X denotes the data matrix
%%%%%%%%%%  r denotes the desired rank
%%%%%%%%%%  thres denotes the convergence threshold, usually 10^-3
%%%%%%%%%%  initial estimate of the noise covariance matrix

M      = size(X,1);      %%%%% dimension
R_cap  = X*X'/size(X,2); %%%%%Sample covariance matrix
Q      = init;           %%%%% Initialization for the diagonal matrix
IT     = 1;              %%%%% loop indicator
count  = 0;              %%%% loop counter
LL     = 10^5;           %%%%Initial value of the log-likelihood, taken to be a large value.
while(IT==1)
    R_tilda                 = diag(sqrt(1./diag(Q)))*R_cap*diag(sqrt(1./diag(Q)));
    [U,sigma]               = eig(R_tilda); %%% Calculation of U
    [~,ind]                 = sort(diag(real(sigma)),'descend');
    B                       = U(:,ind(1:r))*diag(sqrt(max(diag(real(sigma(ind(1:r),ind(1:r))))-1,0)));
    A                       = inv(eye(M)+B*B');
    tmp                     = (A.*R_cap);
    s                       = sqrt(diag(Q));
    %%%%%%%% Inner loop to calculate the diagonal elements of noise
    %%%%%%%% covariance matrix
    for titer  = 1:10  %%%% the loop over the sigma's are run for fixed number of times (10)
        for k = 1:M
            b                = -(sum(tmp(k,:)'./s) - tmp(k,k)/s(k));
            c                = -real(tmp(k,k));
            s(k)             = (-b+sqrt(b^2-4*c))/2;
        end
    end
    count   = count+1;
    LL_new  = real(log(det(sqrtm(Q)*B*(sqrtm(Q)*B)' + Q)) + trace(R_cap*inv(sqrtm(Q)*B*(sqrtm(Q)*B)' + Q))); %%%Calculation of the log-likelihood objective
    if((abs(LL-LL_new)/abs(LL)<=thres)||(count>500)) %%%% Convergence check
        IT = 0;
    end
    Q             = diag(s.^2);   %%%%% Estimate of the noise covarinace matrix
    LL(count)     = LL_new;
end
R  = sqrtm(Q)*B*(sqrtm(Q)*B)' + Q;  %%%%% Estimated covariance matrix
B  = sqrtm(Q)*B;                    %%%%% Estimated Low rank matrix