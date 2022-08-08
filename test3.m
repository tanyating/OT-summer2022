clear;


%% define distributions (samples)

N = 500; % number of samples of x
M = 500; % number of samples of y
d = 2; % dimension
method = 1; % method to compute bw (1: thumb; 2: deep)

if (d==1)
    x = rand(N,1); % samples drawn from rho (N-by-d matrix)
    y = 8 + (10-8).*rand(M,1); % samples drawn from mu (M-by-d matrix)
%     x = mvnrnd(0,1,N); % samples drawn from rho (N-by-d matrix)
%     y = mvnrnd(100,1,M); % samples drawn from mu (M-by-d matrix)
end

if (d==2)
    x = [-10 -0.5] + [20 1].*rand(N,d); % samples drawn from rho (N-by-d matrix)
    y = [-10 -0.5] + [20 1].*rand(M,d); % samples drawn from mu (M-by-d matrix)
%     x = [1 1] + rand(N,d); % samples drawn from rho (N-by-d matrix)
%     y = [5 5] + [2 2].*rand(M,d); % samples drawn from mu (M-by-d matrix)
%     x = mvnrnd([1;1],5.*eye(2),N); % samples drawn from rho (N-by-d matrix)
%     y = mvnrnd([100;10],eye(2),M); % samples drawn from mu (M-by-d matrix)
%     x = mvnrnd([0;0],[20 0;0 1],N); % samples drawn from rho (N-by-d matrix)
    theta = pi/6;
    x = [cos(theta).*x(:,1)+sin(theta).*x(:,2), -sin(theta).*x(:,1)+cos(theta).*x(:,2)];
    y = y+5;
%     y = mvnrnd([0;0],[20 0;0 1],M); % samples drawn from mu (M-by-d matrix)
%     y = [cos(theta).*y(:,1)+sin(theta).*y(:,2), -sin(theta).*y(:,1)+cos(theta).*y(:,2)];
end

%% normalize x and y
% xy = [x;y];
% xy = xy./std(xy);
% x = xy(1:N,:);
% y = xy(N+1:end,:);

%% start transport
z = x; % start the transport with the original samples
w = x; % start with no free transformation

K = 6000; % a threshold number of steps;
MAX_STEP = K + 6000; % maximum steps of grad dc
INNER_STEP = 1; % inner number of grad dc for each set of lambda and bw

eta = zeros(MAX_STEP+1,1);
eta2 = zeros(MAX_STEP+1,1);
eta3 = zeros(MAX_STEP+1,1);
gradLzNorm = zeros(MAX_STEP+1,1);
gradLwNorm = zeros(MAX_STEP+1,1);

eta(1) = 0.1; % initial (small) learning rate wrt z
eta2(1) = 0.1; % initial (small) learning rate wrt w
eta3(1) = 0.1;
lambda = 5e1; % intial regularization parameter wrt z
lambda_final = 5e6; % final regularization parameter (large)
dl = (lambda_final-lambda)/(K); % lambda increment
gamma = 5; % regularization parameter wrt w
gamma_final = 5000; % final regularization parameter (large)
dg = (gamma_final-gamma)/(K); % lambda increment

alpha = 500; % threshold value to update lambda

% a = bw(z,1); % bandwidth for rho_T
% b = bw(y,1); % bandwidth for mu
c = 8; % initial multiplier of bandwidth
dc = (c-1)/(K); % gradual decrease of c

a = c*bw([z;y],method); % use a common, large bandwidth for rho_T and mu
b = a;

afinal = bw(y,1); % final bw for y using rule of thumb
da = (a-afinal)/(K); % gradual decrease of a (if not using rule of thumb to update)

% initial gradients and objective values
[LF,gradLFz] = grad_LF_z(y,z,a,b);
[LC,gradLCz] = grad_LC_z(w,z);
% L = LC + lambda.*LF;
gradLz = gradLCz + lambda.*gradLFz;
gradLzNorm(1) = norm(gradLz);
[LC,gradLCw] = grad_LC_w(w,z);
[LR,gradLRw] = grad_LR_w(w,x);
% L = LC + lambda.*LF + gamma.*LR;
gradLw = gradLCw + gamma.*gradLRw;
gradLwNorm(1) = norm(gradLw);

tol = 1e-10; % grad norm tolerance
eta_tol = 1e-32; % smallest learning rate
i = 0;
plotstep = MAX_STEP; 
figure();
axis equal;

% transport/grad dc steps
while (i<K) %&& norm(gradL)>tol)
    % plot distribution at some steps
    if (mod(i,plotstep)==0 && d==1) % visualize test function
        clf;
        nbins=10;
        histogram(x,nbins,'FaceColor','r','Normalization','probability'); hold on;
        histogram(y,nbins,'FaceColor','b','Normalization','probability');
        histogram(w,nbins,'FaceColor','m','Normalization','probability');
        histogram(z,nbins,'FaceColor','g','Normalization','probability');
        legend('x','y','w','z');
        title(sprintf("Distribution of x, y, w, and z at final step %d (c = %d)",i,c));
        % print loss with fixed lambda and bw
        disp(sprintf('at %d step, LC = %9.5e, LF = %9.5e',i,LC,LF));
        input('Hit <return> to continue  ');
    end
    if (mod(i,plotstep)==0 && d==2) % visualize test function
        clf;
        plot(x(:,1),x(:,2), 'r.', 'Markersize', 5); hold on;
        plot(y(:,1),y(:,2), 'b.', 'Markersize', 5);
        plot(w(:,1),w(:,2), 'm.', 'Markersize', 5);
        plot(z(:,1),z(:,2), 'g.', 'Markersize', 5);
        legend('x','y','w','z');
        title(sprintf("Distribution of x, y, w, and z at step %d (c = %d)",i,c));
        axis equal;
        % print loss with fixed lambda and bw
        disp(sprintf('at %d step, LC = %9.5e, LF = %9.5e',i,LC,LF));
        input('Hit <return> to continue  ');
    end
    
    % projected method to update lambda
    lambda_min = alpha - sum(gradLCz.*gradLFz,'all')/sum(gradLFz.^2,'all');
    if (lambda_min >= lambda && lambda_min <= lambda_final)
        lambda = lambda_min;
    elseif (lambda_min > lambda_final)
        lambda = lambda_final;
    end
%     lambda
    
    % gradient descent with current set of bw and lambda
    k=0;
    while (k<INNER_STEP) %&& norm(gradL)>tol) % inner loop
        % 1) grad dc wrt z
        [z,eta(i+2)] = grad_dc_z(w,y,z,a,b,lambda,eta(i+1),eta_tol);
%         [z,w,eta(i+2)] = grad_dc(x,w,y,z,a,b,lambda,gamma,eta(i+1),eta_tol);
        
        % 1) grad dc wrt w
%         if (i>5)
        [w,eta2(i+2)] = grad_dc_w(w,x,z,gamma,eta2(i+1),eta_tol);
%         [w,eta2(i+2)] = grad_dc_LC_w(w,z,eta2(i+1),eta_tol);
%         [w,eta3(i+2)] = grad_dc_LR_w(w,x,gamma,eta3(i+1),eta_tol);
%         end
        k = k+1;
    end
    

    % update bw and lambda
    a = a-da; % decrese bw
    b = a;
%     c = c-dc; % decrese the multiplier for bw
%     a = c.*bw([z;y],1); % use rule of thumb to update bw
%     %         b-a
%     b = a;
    
%     lambda = lambda+dl; % increase lambda
    
    gamma = gamma+dg; % increase gamma

    % new function and gradient values
    [LF,gradLFz] = grad_LF_z(y,z,a,b);
    [LC,gradLCz] = grad_LC_z(w,z);
    % L = LC + lambda.*LF;
    gradLz = gradLCz + lambda.*gradLFz;
    gradLzNorm(i+2) = norm(gradLz);
    [LC,gradLCw] = grad_LC_w(w,z);
    [LR,gradLRw] = grad_LR_w(w,x);
    % L = LC + lambda.*LF + gamma.*LR;
    gradLw = gradLCw + gamma.*gradLRw;
    gradLwNorm(i+2) = norm(gradLw);
    
%     norm(gradLCw)
%     norm(gradLRw)
    
%     gradLCw
%     eta2(i+2)
%     gamma.*gradLRw

    
    i = i+1;
end

% gamma = 1;
while (i<MAX_STEP) %|| norm(gradL)>tol) % extra steps for final set of lambda and bw
    % plot distribution at some steps
    if (mod(i,plotstep)==0 && d==1) % visualize test function
        clf;
        nbins=10;
        histogram(x,nbins,'FaceColor','r','Normalization','probability'); hold on;
        histogram(y,nbins,'FaceColor','b','Normalization','probability');
        histogram(w,nbins,'FaceColor','m','Normalization','probability');
        histogram(z,nbins,'FaceColor','g','Normalization','probability');
        legend('x','y','w','z');
        title(sprintf("Distribution of x, y, w, and z at final step %d (c = %d)",i,c));
        % print loss with fixed lambda and bw
        disp(sprintf('at %d step, LC = %9.5e, LF = %9.5e',i,LC,LF));
        input('Hit <return> to continue  ');
    end
    if (mod(i,plotstep)==0 && d==2) % visualize test function
        clf;
        plot(x(:,1),x(:,2), 'r.', 'Markersize', 5); hold on;
        plot(y(:,1),y(:,2), 'b.', 'Markersize', 5);
        plot(w(:,1),w(:,2), 'm.', 'Markersize', 5);
        plot(z(:,1),z(:,2), 'g.', 'Markersize', 5);
        legend('x','y','w','z');
        title(sprintf("Distribution of x, y, w, and z at step %d (c = %d)",i,c));
        axis equal;
        % print loss with fixed lambda and bw
        disp(sprintf('at %d step, LC = %9.5e, LF = %9.5e',i,LC,LF));
        input('Hit <return> to continue  ');
    end
    
    lambda_min = alpha - sum(gradLCz.*gradLFz,'all')/sum(gradLFz.^2,'all');
    if (lambda_min >= lambda && lambda_min <= lambda_final)
        lambda = lambda_min;
    elseif (lambda_min > lambda_final)
        lambda = lambda_final;
    end
    
    % gradient descent with current set of bw and lambda
    k=0;
    while (k<INNER_STEP) %&& norm(gradL)>tol) % inner loop
        % 1) grad dc wrt z
        [z,eta(i+2)] = grad_dc_z(w,y,z,a,b,lambda,eta(i+1),eta_tol);
%         [z,w,eta(i+2)] = grad_dc(x,w,y,z,a,b,lambda,gamma,eta(i+1),eta_tol);

        % 2) grad dc wrt w
        [w,eta2(i+2)] = grad_dc_w(w,x,z,gamma,eta2(i+1),eta_tol);
%         [w,eta2(i+2)] = grad_dc_LC_w(w,z,eta2(i+1),eta_tol);
%         [w,eta3(i+2)] = grad_dc_LR_w(w,x,gamma,eta3(i+1),eta_tol);
        
        k = k+1;
    end
    
    
    % new function and gradient values
    [LF,gradLFz] = grad_LF_z(y,z,a,b);
    [LC,gradLCz] = grad_LC_z(w,z);
    % L = LC + lambda.*LF;
    gradLz = gradLCz + lambda.*gradLFz;
    gradLzNorm(i+2) = norm(gradLz);
    [LC,gradLCw] = grad_LC_w(w,z);
    [LR,gradLRw] = grad_LR_w(w,x);
    % L = LC + lambda.*LF + gamma.*LR;
    gradLw = gradLCw + gamma.*gradLRw;
    gradLwNorm(i+2) = norm(gradLw);
%     norm(gradLRw)
%     norm(gradLCw)
    i = i+1;
end


% final L (loss)
disp(sprintf('at %d step, LC = %9.5e, LF = %9.5e',i,LC,LF));
% [LC,gradRot] = grad_rot2d(w,z);
% [LC,gradTr] = grad_tr(w,z);
norm(gradLz)
norm(gradLw)
LR

% visualize results
% figure();

% plot in 1d
if (d==1)
    nbins=10;
    histogram(x,nbins,'FaceColor','r','Normalization','probability'); hold on;
    histogram(y,nbins,'FaceColor','b','Normalization','probability');
    histogram(w,nbins,'FaceColor','m','Normalization','probability');
    histogram(z,nbins,'FaceColor','g','Normalization','probability');
    legend('x','y','w','z');
    title(sprintf("Distribution of x, y, w, and z at final step %d (c = %d)",i,c));
end

% plot in 2d
if (d==2)
    plot(x(:,1),x(:,2), 'r.', 'Markersize', 5); hold on;
    plot(y(:,1),y(:,2), 'b.', 'Markersize', 5);
    plot(w(:,1),w(:,2), 'm.', 'Markersize', 5);
    plot(z(:,1),z(:,2), 'g.', 'Markersize', 5);
    legend('x','y','w','z');
    axis equal;
    title(sprintf("Distribution of x, y, w, and z at final step %d (c = %d)",i,c));
end

% plot learning rates
figure();
plot(0:MAX_STEP,eta,'r.-'); hold on;
xlabel('number of steps');
ylabel('\eta');
title('learning rates for grad dc wrt z');

figure();
plot(0:MAX_STEP,eta2,'b.-');
xlabel('number of steps');
ylabel('\eta');
title('learning rates for grad dc wrt w');


% plot gradient norms
figure();
plot(0:MAX_STEP,gradLzNorm,'r.-');
xlabel('number of steps');
ylabel('gradient norm');
title('gradient norm wrt z');

figure();
plot(0:MAX_STEP,gradLwNorm,'b.-');
xlabel('number of steps');
ylabel('gradient norm');
title('gradient norm wrt w');


%---------------------------------------------

%% Gradient 

% compute L_C and gradient for L_C wrt z
function [res1,res2] = grad_LC_z(w,z)

% res1: L_C
% res2: grad of L_C wrt z

N = length(w(:,1));
tmp = w-z;
res1 = 1/N/2.*sum(tmp.^2,'all');
res2 = -1/N.*(tmp);

end


% compute L_F and gradient for L_F (without lambda) wrt z
function [res1,res2] = grad_LF_z(y,z,a,b)

N = length(z(:,1));
M = length(y(:,1));
d = length(z(1,:));

% approach 1 for grad: memory costly, but faster?

% store all pairs of z-z and y-z
tmp1 = zeros(d,N,N);
tmp2 = zeros(d,N,M);
tmp3 = zeros(d,M,N);
tmp4 = zeros(d,M,M);

for l=1:d
     tmp1(l,:,:) = (z(:,l)' - z(:,l))./a(l);
     tmp2(l,:,:) = (y(:,l)' - z(:,l))./b(l);
     tmp3(l,:,:) = (z(:,l)' - y(:,l))./a(l);
     tmp4(l,:,:) = (y(:,l)' - y(:,l))./b(l);
end

tmp5 = (sum(tmp1.*exp(-1/2.*sum(tmp1.^2,1)),3));
tmp6 = (sum(tmp2.*exp(-1/2.*sum(tmp2.^2,1)),3));

% normalizing constants
c1 = 1/(N^2)/(prod(a)*(sqrt(2*pi))^d);
c2 = 1/(M*N)/(prod(b)*(sqrt(2*pi))^d);
c3 = 1/(M*N)/(prod(a)*(sqrt(2*pi))^d);
c4 = 1/(M^2)/(prod(b)*(sqrt(2*pi))^d);

res2 = c1./(a).*(tmp5)' - c2./(b).*(tmp6)'; 

res1 = c1.*sum(exp(-1/2.*sum(tmp1.^2,1)),'all') - c2.*sum(exp(-1/2.*sum(tmp2.^2,1)),'all')...
    - c3.*sum(exp(-1/2.*sum(tmp3.^2,1)),'all') + c4.*sum(exp(-1/2.*sum(tmp4.^2,1)),'all');



end

% compute L_C and gradient for L_C wrt w
function [res1,res2] = grad_LC_w(w,z)

% res1: L_C
% res2: grad of L_C wrt z

N = length(w(:,1));
tmp = w-z;
res1 = 1/N/2.*sum(tmp.^2,'all');
res2 = 1/N.*(tmp);

end


% compute L_R and gradient for L_R (without gamma) wrt w
function [res1,res2] = grad_LR_w(w,x)

N = length(w(:,1));
d = length(w(1,:));
eps_sqr = var(x,0,'all')/1000; % some small number

% approach 1 for grad: memory costly, but faster?

% store all pairs of x-x and w-w
tmp1 = zeros(d,N,N);
tmp2 = zeros(d,N,N);

for l=1:d
     tmp1(l,:,:) = (w(:,l) - w(:,l)');
     tmp2(l,:,:) = (x(:,l) - x(:,l)');
end

tmp3 = sum(tmp1.^2,1);
tmp4 = sum(tmp2.^2,1);
tmp5 = (tmp3./(tmp4+eps_sqr)-1);
tmp6 = tmp1./(tmp4+eps_sqr);

% tmp7 = reshape(tmp5.^2,N,N);
% diag(tmp7)
% tmp7(N,N/2)
% tmp7(N/2,N)

res1 = (sum((tmp5.^2)./2,'all'));
res1 = res1'./(N^2);

res2 = (sum(4*tmp5.*tmp6,3));%./2; 
res2 = res2'./(N^2);

end


%% KDE bandwidths (default Gaussian kernel)

% deep method (negative) gradient wrt a (max log-likelihood)
function [res1,res2] = grad_bw(x,a)

% x: N samples (data)
% a: bandwidth
% res1: (-) likelihood
% res2: (-) grad of log-likelihood wrt a

N = length(x(:,1));
d = length(x(1,:));
res2 = -N;

tmp1 = zeros(d,N,N);

for l=1:d
     tmp1(l,:,:) = ((x(:,l)' - x(:,l))./a).^2;
end

% tmp = exp(-1/2.*sum(tmp1,1));
% tmp = reshape(tmp,[N,N]);
% any(diag(tmp)~=1,'all')

tmp2 = (sum(sum(tmp1,1).*exp(-1/2.*sum(tmp1,1)),3)); % numerator
tmp3 = (sum(exp(-1/2.*sum(tmp1,1)),3)) - 1; % denominator

res2 = res2 + sum(tmp2./tmp3);
res2 = -res2/a;

% tmp3 = tmp3.*100;
% 1/(N-1)/((a*sqrt(2*pi))^d)
% any(1/(N-1)/((a*sqrt(2*pi))^d).*tmp3<0,'all')
res1 = -sum(log(1/(N-1)/((a*sqrt(2*pi))^d).*tmp3));


end

function [a] = bw(x,method)

N = length(x(:,1));
d = length(x(1,:));

% Rule of Thumb
% a = (4/(d+2))^(1/(d+4))*(N^(-1/(d+4)))*mean(std(x));
a = (4/(d+2))^(1/(d+4))*(N^(-1/(d+4))).*std(x);
if (method == 1)
    return;

elseif (method == 2)
    % deep method (gradient ascent)
%     a = a*10; % initial guess
    max_steps = 10; % maximum number of steps
    eta = 0.1; % initial learning rate
    
    [P,grada] = grad_bw(x,a);
%     P
    
    tol = 1e-2; % (relative) tolerance
    i = 0;
    while (i<max_steps && norm(grada)>tol)
        eta = eta*2;
        anew = a - eta.*grada;
        [Pnew,grada_new] = grad_bw(x,anew);
        while ((Pnew > P || anew<0) && eta > 1e-32)%|| anew < 0) %&& (abs(L-Lnew)>0.1))
            eta = eta/2;
            anew = a - eta.*grada;
            [Pnew,grada_new] = grad_bw(x,anew);
        end

        if (anew>0 && Pnew < P)
            a = anew;
            P = Pnew;
            grada = grada_new;
        end
        i = i+1;
%         norm(grada)
    end

end

end

% % Rule of Thumb
% bw1 = @(x,N)(0.9*min(std(x),iqr(x)/1.34)*(N^(-1/5)));

%% Gradient Descent (one step)

% grad dc wrt z and w together
function [znew,wnew,eta] = grad_dc(x,w,y,z,a,b,lambda,gamma,eta,eta_tol)

% lambda: regularization parameter
% eta: learning rate
% eta_tol: tolerance for (smallest) learning rate

[LF,gradLF] = grad_LF_z(y,z,a,b);
[LC,gradLCz] = grad_LC_z(w,z);
[LR,gradLR] = grad_LR_w(w,x);
[LC,gradLCw] = grad_LC_w(w,z);

L = LC + lambda.*LF + gamma.*LR;

eta = eta*2;
gradLz = gradLCz + lambda.*gradLF;
znew = z - eta.*gradLz;
gradLw = gradLCw + gamma.*gradLR;
wnew = w - eta.*gradLw;

[LFnew,gradLF_new] = grad_LF_z(y,znew,a,b);
[LCnew,gradLCz_new] = grad_LC_z(wnew,znew);
[LRnew,gradLR_new] = grad_LR_w(wnew,x);
[LCnew,gradLCw_new] = grad_LC_w(wnew,znew);

% k=0;
Lnew = LCnew + lambda.*LFnew + gamma.*LRnew;

while (Lnew > L && eta>eta_tol) %&& (abs(L-Lnew)>0.1))
% while ((n2 >= n1) && (abs(n1-n2)>0.1)) % find reasonable learning rate (when the gradient is closer to 0)
    eta = eta/2;
    znew = z - eta.*gradLz;
    wnew = w - eta.*gradLw;
    [LFnew,gradLF_new] = grad_LF_z(y,znew,a,b);
    [LCnew,gradLCz_new] = grad_LC_z(wnew,znew);
    [LRnew,gradLR_new] = grad_LR_w(wnew,x);
    [LCnew,gradLCw_new] = grad_LC_w(wnew,znew);
    Lnew = LCnew + lambda.*LFnew + gamma.*LRnew;

end

% gradL_new = gradLC_new + lambda.*gradLF_new;
if (Lnew > L) % if no decrease in objective, don't descent
    znew = z;
    wnew = w;
%     Lnew = L;
%     LCnew = LC;
%     LFnew = LF;
%     gradLC_new = gradLC;
%     gradLF_new = gradLF;
end

end

% grad dc wrt z
function [znew,eta] = grad_dc_z(w,y,z,a,b,lambda,eta,eta_tol)

% lambda: regularization parameter
% eta: learning rate
% gradL: gradient of the objective L wrt current z
% eta_tol: tolerance for (smallest) learning rate

[LF,gradLF] = grad_LF_z(y,z,a,b);
[LC,gradLC] = grad_LC_z(w,z);

eta = eta*2;
gradL = gradLC + lambda.*gradLF;
znew = z - eta.*gradL;
[LFnew,gradLF_new] = grad_LF_z(y,znew,a,b);
[LCnew,gradLC_new] = grad_LC_z(w,znew);


% k=0;
Lnew = LCnew + lambda.*LFnew;
L = LC + lambda.*LF;
while (Lnew > L && eta>eta_tol) %&& (abs(L-Lnew)>0.1))
% while ((n2 >= n1) && (abs(n1-n2)>0.1)) % find reasonable learning rate (when the gradient is closer to 0)
    eta = eta/2;
    znew = z - eta.*gradL;
    [LFnew,gradLF_new] = grad_LF_z(y,znew,a,b);
    [LCnew,gradLC_new] = grad_LC_z(w,znew);
    Lnew = LCnew + lambda.*LFnew;

end

% gradL_new = gradLC_new + lambda.*gradLF_new;
if (Lnew > L) % if no decrease in objective, don't descent
    znew = z;
%     Lnew = L;
%     LCnew = LC;
%     LFnew = LF;
%     gradLC_new = gradLC;
%     gradLF_new = gradLF;
end

end


% grad dc wrt w
function [wnew,eta] = grad_dc_w(w,x,z,gamma,eta,eta_tol)

% gamma: regularization parameter
% eta: learning rate
% gradL: gradient of the objective L wrt current w
% eta_tol: tolerance for (smallest) learning rate

[LR,gradLR] = grad_LR_w(w,x);
[LC,gradLC] = grad_LC_w(w,z);

eta = eta*2;
gradL = gradLC + gamma.*gradLR;
wnew = w - eta.*gradL;
[LRnew,gradLR_new] = grad_LR_w(wnew,x);
[LCnew,gradLC_new] = grad_LC_w(wnew,z);


% k=0;
Lnew = LCnew + gamma.*LRnew;
L = LC + gamma.*LR;
while (Lnew > L && eta>eta_tol) %&& (abs(L-Lnew)>0.1))
% while ((n2 >= n1) && (abs(n1-n2)>0.1)) % find reasonable learning rate (when the gradient is closer to 0)
    eta = eta/2;
    wnew = w - eta.*gradL;
    [LRnew,gradLR_new] = grad_LR_w(wnew,x);
    [LCnew,gradLC_new] = grad_LC_w(wnew,z);
    Lnew = LCnew + gamma.*LRnew;

end

% gradL_new = gradLC_new + lambda.*gradLF_new;
if (Lnew > L) % if no decrease in objective, don't descent
    wnew = w;
%     Lnew = L;
%     LCnew = LC;
%     LFnew = LF;
%     gradLC_new = gradLC;
%     gradLF_new = gradLF;

% else
%     eta.*gradL
%     eta
    
end

end


% gradient descent LR wrt w
function [wnew,eta] = grad_dc_LR_w(w,x,gamma,eta,eta_tol)

% eta: learning rate
% eta_tol: tolerance for (smallest) learning rate

eta = eta*2;
[LR,gradLR] = grad_LR_w(w,x);
wnew = w - eta.*gamma*gradLR;

[LRnew,gradLR_new] = grad_LR_w(wnew,x);
while (LRnew > LR && eta>eta_tol) %&& (abs(L-Lnew)>0.1))
% while ((n2 >= n1) && (abs(n1-n2)>0.1)) % find reasonable learning rate (when the gradient is closer to 0)
    eta = eta/2;
    wnew = w - eta.*gamma*gradLR;
    [LRnew,gradLR_new] = grad_LR_w(wnew,x);

end

% gradL_new = gradLC_new + lambda.*gradLF_new;
if (LRnew > LR) % if no decrease in objective, don't descent
    wnew = w;
%     LCnew = LC;
end

end

% gradient descent LC wrt w
function [wnew,eta] = grad_dc_LC_w(w,z,eta,eta_tol)

% eta: learning rate
% eta_tol: tolerance for (smallest) learning rate

eta = eta*2;
[LC,gradLC] = grad_LC_w(w,z);
wnew = w - eta.*gradLC;

[LCnew,gradLC_new] = grad_LC_w(wnew,z);
while (LCnew > LC && eta>eta_tol) %&& (abs(L-Lnew)>0.1))
% while ((n2 >= n1) && (abs(n1-n2)>0.1)) % find reasonable learning rate (when the gradient is closer to 0)
    eta = eta/2;
    wnew = w - eta.*gradLC;
    [LCnew,gradLC_new] = grad_LC_w(wnew,z);

end

% gradL_new = gradLC_new + lambda.*gradLF_new;
if (LCnew > LC) % if no decrease in objective, don't descent
    wnew = w;
%     LCnew = LC;
end

end