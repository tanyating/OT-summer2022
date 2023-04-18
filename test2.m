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
% %     y = y+0;
%     y = mvnrnd([0;0],[20 0;0 1],M); % samples drawn from mu (M-by-d matrix)
end

%% normalize x and y
% xy = [x;y];
% xy = xy./std(xy);
% x = xy(1:N,:);
% y = xy(N+1:end,:);

%% start transport
z = x; % start the transport with the original samples
w = x; % start with no free transformation

K = 500; % a threshold number of steps;
MAX_STEP = K + 500; % maximum steps of grad dc
INNER_STEP = 1; % inner number of grad dc for each set of lambda and bw

eta = zeros(MAX_STEP+1,1);
eta2 = zeros(MAX_STEP+1,1);
eta3 = zeros(MAX_STEP+1,1);
gradLNorm = zeros(MAX_STEP+1,1);
gradTrNorm = zeros(MAX_STEP+1,1);
gradRotNorm = zeros(MAX_STEP+1,1);

eta(1) = 0.1; % initial (small) learning rate wrt z
eta2(1) = 0.1; % initial (small) learning rate wrt free trans
eta3(1) = 0.1; % initial (small) learning rate wrt free rot
lambda = 5e2; % intial regularization parameter 
lambda_final = 5e4; % final regularization parameter (large)
dl = (lambda_final-lambda)/(K); % lambda increment

% a = bw(z,1); % bandwidth for rho_T
% b = bw(y,1); % bandwidth for mu
c = 8; % initial multiplier of bandwidth
dc = (c-1)/(K); % gradual decrease of c

a = c*bw([z;y],method); % use a common, large bandwidth for rho_T and mu
b = a;

afinal = bw(y,1); % final bw for y using rule of thumb
da = (a-afinal)/(K); % gradual decrease of a (if not using rule of thumb to update)

% initial gradients and objective values
zc = z;
[LF,gradLF] = grad_LF(y,z,zc,a,b);
[LC,gradLC] = grad_LC(w,z);
% L = LC + lambda.*LF;
gradL = gradLC + lambda.*gradLF;
gradLNorm(1) = norm(gradL);
[LC,gradRot] = grad_rot2d(w,z);
[LC,gradTr] = grad_tr(w,z);
gradTrNorm(1) = norm(gradTr);
gradRotNorm(1) = norm(gradRot);

tol = 1e-10; % grad norm tolerance
eta_tol = 1e-32; % smallest learning rate
i = 0;
plotstep = MAX_STEP; 
figure();

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
        % print loss with fixed lambda and bw
        disp(sprintf('at %d step, LC = %9.5e, LF = %9.5e',i,LC,LF));
        input('Hit <return> to continue  ');
    end
    
    % gradient descent with current set of bw and lambda
    k=0;
    while (k<INNER_STEP && norm(gradL)>tol) % inner loop
        % 1) grad dc wrt z
        [z,zc,eta(i+2)] = grad_dc_z(w,y,z,zc,a,b,lambda,eta(i+1),eta_tol);
%         [z,w,eta(i+2)] = grad_dc(w,y,z,a,b,lambda,eta(i+1),eta_tol);
        
        % 2) grad dc wrt w
        [w,eta3(i+2)] = grad_dc_rot2d(w,z,eta3(i+1),eta_tol);
%         [w,eta2(i+2)] = grad_dc_tr(w,z,eta2(i+1),eta_tol);
%         [w,eta2(i+2)] = grad_dc_rot_tr(w,z,eta2(i+1),eta_tol);

        
        k = k+1;
    end
    

    % update bw and lambda
    a = a-da; % decrese bw
    b = a;
    
    lambda = lambda+dl; % increase lambda

    % new function and gradient values
    [LF,gradLF] = grad_LF(y,z,zc,a,b);
    [LC,gradLC] = grad_LC(w,z);
%     L = LC + lambda.*LF;
    gradL = gradLC + lambda.*gradLF;
    gradLNorm(i+2) = norm(gradL);
    [LC,gradRot] = grad_rot2d(w,z);
    [LC,gradTr] = grad_tr(w,z);
    gradTrNorm(i+2) = norm(gradTr);
    gradRotNorm(i+2) = norm(gradRot);
    
    
    i = i+1;
end

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
        % print loss with fixed lambda and bw
        disp(sprintf('at %d step, LC = %9.5e, LF = %9.5e',i,LC,LF));
        input('Hit <return> to continue  ');
    end
    
     % gradient descent with current set of bw and lambda   
    k=0;
    while (k<INNER_STEP && norm(gradL)>tol) % inner loop
        % 1) grad dc wrt z
        [z,zc,eta(i+2)] = grad_dc_z(w,y,z,zc,a,b,lambda,eta(i+1),eta_tol);
%         [z,w,eta(i+2)] = grad_dc(w,y,z,a,b,lambda,eta(i+1),eta_tol);
        
        % 2) grad dc wrt w
        [w,eta3(i+2)] = grad_dc_rot2d(w,z,eta3(i+1),eta_tol);
%         [w,eta2(i+2)] = grad_dc_tr(w,z,eta2(i+1),eta_tol);
%         [w,eta2(i+2)] = grad_dc_rot_tr(w,z,eta2(i+1),eta_tol);
        
        
        k = k+1;
    end
    
    % new function and gradient values
    [LF,gradLF] = grad_LF(y,z,zc,a,b);
    [LC,gradLC] = grad_LC(w,z);
%     L = LC + lambda.*LF;
    gradL = gradLC + lambda.*gradLF;
    gradLNorm(i+2) = norm(gradL);
    [LC,gradRot] = grad_rot2d(w,z);
    [LC,gradTr] = grad_tr(w,z);
    gradTrNorm(i+2) = norm(gradTr);
    gradRotNorm(i+2) = norm(gradRot);
    
    i = i+1;
end


% final L (loss)
disp(sprintf('at %d step, LC = %9.5e, LF = %9.5e',i,LC,LF));
% [LC,gradRot] = grad_rot2d(w,z);
% [LC,gradTr] = grad_tr(w,z);
norm(gradL)
norm(gradTr)
norm(gradRot)
LC
LF

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
    title(sprintf("Distribution of x, y, w, and z at final step %d (c = %d)",i,c));
end

% plot learning rates
figure();
plot(0:MAX_STEP,eta,'r.-'); hold on;
% plot(0:MAX_STEP,eta2,'b.', 'Markersize', 3);
% plot(0:MAX_STEP,eta3,'g.', 'Markersize', 3);
xlabel('number of steps');
ylabel('\eta');
% legend('grad dc wrt z','free translation','free rotation');
% title('learning rates');
title('learning rates for grad dc wrt z');

figure();
plot(0:MAX_STEP,eta2,'b.-');
xlabel('number of steps');
ylabel('\eta');
title('learning rates for grad dc wrt free translation');

figure();
plot(0:MAX_STEP,eta3,'g.-');
xlabel('number of steps');
ylabel('\eta');
title('learning rates for grad dc wrt free rotation');

% plot gradient norms
figure();
plot(0:MAX_STEP,gradLNorm,'r.-');
xlabel('number of steps');
ylabel('gradient norm');
title('gradient norm wrt z');

figure();
plot(0:MAX_STEP,gradTrNorm,'b.-');
xlabel('number of steps');
ylabel('gradient norm');
title('gradient norm wrt free translation');

figure();
plot(0:MAX_STEP,gradRotNorm,'g.-');
xlabel('number of steps');
ylabel('gradient norm');
title('gradient norm wrt free rotation');

%---------------------------------------------

%% Gradient 

% compute L_C and gradient for L_C
function [res1,res2] = grad_LC(x,z)

% res1: L_C
% res2: grad of L_C wrt z

N = length(x(:,1));
tmp = x-z;
res1 = 1/N/2.*sum(tmp.^2,'all');
res2 = -1/N.*(tmp);

end

% compute L_C and gradient for translation
function [res1,res2] = grad_tr(w,z)

% res1: L_C
% res2: grad of L_C wrt delta=0

N = length(w(:,1));
tmp = w-z;
res1 = 1/N/2.*sum(tmp.^2,'all');
res2 = 1/N.*sum(tmp,1);

end

% compute L_C and gradient for rotation in 2D
function [res1,res2] = grad_rot2d(w,z)

% res1: L_C
% res2: grad of L_C wrt theta=0

N = length(w(:,1));
tmp = w-z;
res1 = 1/N/2.*sum(tmp.^2,'all');
% res2 = 1/N.*sum(tmp(:,1).*w(:,2)-tmp(:,2).*w(:,1));
res2 = sum(-z(:,1).*w(:,2) + z(:,2).*w(:,1))/N;

end

% compute hessian (2nd derivative) for rotation in 2D
function [res] = hess_rot2d(w,z)

% res: 2nd derivative of L_C wrt theta=0

N = length(w(:,1));
res = sum(z(:,1).*w(:,1) + z(:,2).*w(:,2))/N;

end

% compute test function F at y and z
function [Fz,Fy] = F(y,z,a,b)

N = length(z(:,1));
M = length(y(:,1));
d = length(z(1,:));

tmp1 = zeros(d,N,N);
tmp2 = zeros(d,N,M);
tmp3 = zeros(d,M,N);
tmp4 = zeros(d,M,M);

for l=1:d
     tmp1(l,:,:) = (z(:,l)' - z(:,l))./a;
     tmp2(l,:,:) = (y(:,l)' - z(:,l))./b;
     tmp3(l,:,:) = (z(:,l)' - y(:,l))./a;
     tmp4(l,:,:) = (y(:,l)' - y(:,l))./b;
end

% normalizing constants
c1 = 1/(N)/((a*sqrt(2*pi))^d);
c2 = 1/(M)/((b*sqrt(2*pi))^d);

Fz = c1.*sum(exp(-1/2.*sum(tmp1.^2,1)),3) - c2.*sum(exp(-1/2.*sum(tmp2.^2,1)),3);
Fy = c1.*sum(exp(-1/2.*sum(tmp3.^2,1)),3) - c2.*sum(exp(-1/2.*sum(tmp4.^2,1)),3);

Fz = Fz';
Fy = Fy';

end

% compute L_F and gradient for L_F (without lambda)
function [res1,res2] = grad_LF(y,z,c,a,b)

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
     tmp1(l,:,:) = (c(:,l)' - z(:,l))./a(l);
     tmp2(l,:,:) = (y(:,l)' - z(:,l))./b(l);
     tmp3(l,:,:) = (c(:,l)' - y(:,l))./a(l);
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
% grad dc wrt z and rot/tr (w)
function [znew,wnew,eta] = grad_dc(w,y,z,a,b,lambda,eta,eta_tol)

% lambda: regularization parameter
% eta: learning rate
% eta_tol: tolerance for (smallest) learning rate

[LF,gradLF] = grad_LF(y,z,a,b);
[LC,gradLC] = grad_LC(w,z);
gradL = gradLC + lambda.*gradLF;
[LC,gradRot] = grad_rot2d(w,z);
[LC,gradTr] = grad_tr(w,z);
L = LC + lambda.*LF;


eta = eta*2;
znew = z - eta.*gradL;
theta = -eta*gradRot;
delta = -eta.*gradTr;
wnew = [cos(theta).*w(:,1)+sin(theta).*w(:,2), -sin(theta).*w(:,1)+cos(theta).*w(:,2)] + delta;

[LFnew,gradLF_new] = grad_LF(y,znew,a,b);
[LCnew,gradLC_new] = grad_LC(wnew,znew);
Lnew = LCnew + lambda.*LFnew;


while (Lnew > L && eta>eta_tol) %&& (abs(L-Lnew)>0.1))
% while ((n2 >= n1) && (abs(n1-n2)>0.1)) % find reasonable learning rate (when the gradient is closer to 0)
    eta = eta/2;
    znew = z - eta.*gradL;
    theta = -eta*gradRot;
    delta = -eta.*gradTr;
    wnew = [cos(theta).*w(:,1)+sin(theta).*w(:,2), -sin(theta).*w(:,1)+cos(theta).*w(:,2)] + delta;
    
    [LFnew,gradLF_new] = grad_LF(y,znew,a,b);
    [LCnew,gradLC_new] = grad_LC(wnew,znew);
    Lnew = LCnew + lambda.*LFnew;

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
function [znew,cnew,eta] = grad_dc_z(x,y,z,c,a,b,lambda,eta,eta_tol)

% lambda: regularization parameter
% eta: learning rate
% gradL: gradient of the objective L wrt current z
% eta_tol: tolerance for (smallest) learning rate

[LF,gradLF] = grad_LF(y,z,c,a,b);
[LC,gradLC] = grad_LC(x,z);

eta = eta*2;
gradL = gradLC + lambda.*gradLF;
znew = z - eta.*gradL;
cnew = znew;
[LFnew,gradLF_new] = grad_LF(y,znew,cnew,a,b);
[LCnew,gradLC_new] = grad_LC(x,znew);


% k=0;
Lnew = LCnew + lambda.*LFnew;

[LF,gradLF] = grad_LF(y,z,cnew,a,b);
% [LC,gradLC] = grad_LC(x,z);
L = LC + lambda.*LF;
while (Lnew > L && eta>eta_tol) %&& (abs(L-Lnew)>0.1))
% while ((n2 >= n1) && (abs(n1-n2)>0.1)) % find reasonable learning rate (when the gradient is closer to 0)
    eta = eta/2;
    znew = z - eta.*gradL;
    cnew = znew;
    [LFnew,gradLF_new] = grad_LF(y,znew,cnew,a,b);
    [LCnew,gradLC_new] = grad_LC(x,znew);
    Lnew = LCnew + lambda.*LFnew;
    
    [LF,gradLF] = grad_LF(y,z,cnew,a,b);
    % [LC,gradLC] = grad_LC(x,z);
    L = LC + lambda.*LF;

end

% gradL_new = gradLC_new + lambda.*gradLF_new;
if (Lnew > L) % if no decrease in objective, don't descent
    znew = z;
    cnew = c;
%     Lnew = L;
%     LCnew = LC;
%     LFnew = LF;
%     gradLC_new = gradLC;
%     gradLF_new = gradLF;
end

end

% gradient descent for free translation (wrt delta=0)
function [wnew,eta] = grad_dc_tr(w,z,eta,eta_tol)

% eta: learning rate
% eta_tol: tolerance for (smallest) learning rate

eta = eta*2;
[LC,gradTr] = grad_tr(w,z);
delta = -eta.*gradTr;
wnew = w + delta;

[LCnew,gradTr_new] = grad_tr(wnew,z);
while (LCnew > LC && eta>eta_tol) %&& (abs(L-Lnew)>0.1))
% while ((n2 >= n1) && (abs(n1-n2)>0.1)) % find reasonable learning rate (when the gradient is closer to 0)
    eta = eta/2;
    delta = -eta.*gradTr;
%     size(delta)
    wnew = w + delta;
    [LCnew,gradTr_new] = grad_tr(wnew,z);

end

% gradL_new = gradLC_new + lambda.*gradLF_new;
if (LCnew > LC) % if no decrease in objective, don't descent
    wnew = w;
%     LCnew = LC;
end

end

% gradient descent for free rotation and translation (wrt theta,delta=0)
function [wnew,eta] = grad_dc_rot_tr(w,z,eta,eta_tol)

% eta: learning rate
% eta_tol: tolerance for (smallest) learning rate

eta = eta*2;
[LC,gradRot] = grad_rot2d(w,z);
theta = -eta*gradRot;
[LC,gradTr] = grad_tr(w,z);
delta = -eta.*gradTr;
wnew = [cos(theta).*w(:,1)+sin(theta).*w(:,2), -sin(theta).*w(:,1)+cos(theta).*w(:,2)] + delta;
% wnew = wnew + delta;

[LCnew,gradRot_new] = grad_rot2d(wnew,z);
while (LCnew > LC && eta>eta_tol) %&& (abs(L-Lnew)>0.1))
% while ((n2 >= n1) && (abs(n1-n2)>0.1)) % find reasonable learning rate (when the gradient is closer to 0)
    eta = eta/2;
    theta = -eta*gradRot;
    delta = -eta.*gradTr;
%     size(delta)
    wnew = [cos(theta).*w(:,1)+sin(theta).*w(:,2), -sin(theta).*w(:,1)+cos(theta).*w(:,2)] + delta;
%     wnew = wnew + delta;
    [LCnew,gradRot_new] = grad_rot2d(wnew,z);

end

% gradL_new = gradLC_new + lambda.*gradLF_new;
if (LCnew > LC) % if no decrease in objective, don't descent
    wnew = w;
%     LCnew = LC;
end

end

% gradient descent for free rotation (wrt theta=0)
function [wnew,eta] = grad_dc_rot2d(w,z,eta,eta_tol)

% eta: learning rate
% eta_tol: tolerance for (smallest) learning rate

eta = eta*2;
[LC,gradRot] = grad_rot2d(w,z);
theta = -eta*gradRot;
wnew = [cos(theta).*w(:,1)+sin(theta).*w(:,2), -sin(theta).*w(:,1)+cos(theta).*w(:,2)];

[LCnew,gradRot_new] = grad_rot2d(wnew,z);
while (LCnew > LC && eta>eta_tol) %&& (abs(L-Lnew)>0.1))
% while ((n2 >= n1) && (abs(n1-n2)>0.1)) % find reasonable learning rate (when the gradient is closer to 0)
    eta = eta/2;
    theta = -eta*gradRot;
%     size(delta)
    wnew = [cos(theta).*w(:,1)+sin(theta).*w(:,2), -sin(theta).*w(:,1)+cos(theta).*w(:,2)];
    [LCnew,gradRot_new] = grad_rot2d(wnew,z);

end

% gradL_new = gradLC_new + lambda.*gradLF_new;
if (LCnew > LC) % if no decrease in objective, don't descent
    wnew = w;
%     LCnew = LC;
% else
%     eta
end

end



%% implicit gradient descent

% implicit gradient descent for free translation (wrt delta=0)
function [wnew,eta] = imp_grad_dc_tr(w,z,
zeta,eta_tol)

% eta: learning rate
% eta_tol: tolerance for (smallest) learning rate

eta = eta*2;
[LC,gradTr] = grad_tr(w,z);
delta = -eta/(1+eta).*gradTr;
wnew = w + delta;

[LCnew,gradTr_new] = grad_tr(wnew,z);
while (LCnew > LC && eta>eta_tol) %&& (abs(L-Lnew)>0.1))
% while ((n2 >= n1) && (abs(n1-n2)>0.1)) % find reasonable learning rate (when the gradient is closer to 0)
    eta = eta/2;
    delta = -eta/(1+eta).*gradTr;
%     size(delta)
    wnew = w + delta;
    [LCnew,gradTr_new] = grad_tr(wnew,z);

end

% gradL_new = gradLC_new + lambda.*gradLF_new;
if (LCnew > LC) % if no decrease in objective, don't descent
    wnew = w;
%     LCnew = LC;
end

end


% implicit gradient descent for free rotation (wrt theta=0)
function [wnew,eta] = imp_grad_dc_rot2d(w,z,eta,eta_tol)

% eta: learning rate
% eta_tol: tolerance for (smallest) learning rate

if (eta<1e32)
eta = eta*2;
end
[LC,gradRot] = grad_rot2d(w,z);
hessRot = hess_rot2d(w,z);
theta = -eta*gradRot/(1+eta*hessRot);
wnew = [cos(theta).*w(:,1)+sin(theta).*w(:,2), -sin(theta).*w(:,1)+cos(theta).*w(:,2)];

[LCnew,gradRot_new] = grad_rot2d(wnew,z);
while (LCnew > LC && eta>eta_tol) %&& (abs(L-Lnew)>0.1))
% while ((n2 >= n1) && (abs(n1-n2)>0.1)) % find reasonable learning rate (when the gradient is closer to 0)
    eta = eta/2;
    theta = -eta*gradRot/(1+eta*hessRot);
%     size(delta)
    wnew = [cos(theta).*w(:,1)+sin(theta).*w(:,2), -sin(theta).*w(:,1)+cos(theta).*w(:,2)];
    [LCnew,gradRot_new] = grad_rot2d(wnew,z);

end

% gradL_new = gradLC_new + lambda.*gradLF_new;
if (LCnew > LC) % if no decrease in objective, don't descent
    wnew = w;
%     LCnew = LC;
% else
%     eta/(1+eta*hessRot)
end

end