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
%     x = [1 1] + rand(N,d); % samples drawn from rho (N-by-d matrix)
%     y = [5 5] + [2 2].*rand(M,d); % samples drawn from mu (M-by-d matrix)
    x = mvnrnd([1;1],5.*eye(2),N); % samples drawn from rho (N-by-d matrix)
    y = mvnrnd([100;10],eye(2),M); % samples drawn from mu (M-by-d matrix)
end

% normalize x and y
xy = [x;y];
xy = xy./std(xy);
x = xy(1:N,:);
y = xy(N+1:end,:);

z = x; % start the transport with the original samples

eta = 0.1; % initial (small) learning rate
gamma = 100; % a threshold number of steps;
MAX_STEP = gamma + 10; % maximum steps of grad dc
INNER_STEP = 3; % inner number of grad dc for each set of lambda and bw

lambda = 5e3; % intial regularization parameter 
lambda_final = 5e5; % final regularization parameter (large)
dl = (lambda_final-lambda)/(gamma); % lambda increment

% a = bw(z,1); % bandwidth for rho_T
% b = bw(y,1); % bandwidth for mu
c = 8; % initial multiplier of bandwidth
dc = (c-1)/(gamma); % gradual decrease of c

a = c*bw([z;y],method); % use a common, large bandwidth for rho_T and mu
b = a;

afinal = bw(y,1); % final bw for y using rule of thumb
da = (a-afinal)/(gamma); % gradual decrease of a (if not using rule of thumb to update)

[LF,gradLF] = grad_LF(y,z,a,b);
[LC,gradLC] = grad_LC(x,z);
L = LC + lambda.*LF;
gradL = gradLC + lambda.*gradLF;
% gradL0 = gradL;
% gradL0 = grad_LC(x,z) + lambda.*grad_LF(y,z,a,b); % initial gradient wrt z
% gradL = gradL0;

% plot test functions with initial bw
if (d==1) % visualize test function
    [Fz,Fy] = F(y,z,a,b);
    figure();
    plot(z,Fz,'.','Markersize',1);
    hold on;
    plot(y,Fy,'.','Markersize',1);
    ylim([-1,1]);
    legend('F(z)','F(y)');
    title(sprintf("F evaluated at step 0 (with bandwidth multiplier %d)",c));
end
if (d==2) % visualize test function
    [Fz,Fy] = F(y,z,a,b);
    figure();
    plot3(z(:,1),z(:,2),Fz,'.','Markersize',1);
    hold on;
    plot3(y(:,1),y(:,2),Fy,'.','Markersize',1);
    zlim([-0.5,0.5]);
    legend('F(z)','F(y)');
    title(sprintf("F evaluated at step 0 (with bandwidth multiplier %d)",c));
end

tol = 1e-3; % grad norm tolerance
eta_tol = 1e-32; % smallest learning rate
i = 0;
plotstep = 5;%MAX_STEP; 
figure();
while (i<MAX_STEP && norm(gradL)>tol)
    % plot distribution at some steps
    if (mod(i,plotstep)==0 && d==1) % visualize test function
        clf;
        nbins=10;
        histogram(x,nbins,'FaceColor','r','Normalization','probability'); hold on;
        histogram(y,nbins,'FaceColor','b','Normalization','probability');
        histogram(z,nbins,'FaceColor','g','Normalization','probability');
        legend('x','y','z');
        title(sprintf("Distribution of x, y, and z at final step %d (c = %d)",i,c));
        % print loss with fixed lambda and bw
        disp(sprintf('at %d step, LC = %9.5e, LF = %9.5e',i,LC,LF));

        if i<MAX_STEP
            input('Hit <return> to continue  '); 
        end
    end
    if (mod(i,plotstep)==0 && d==2) % visualize test function
        clf;
        plot(x(:,1),x(:,2), 'r.', 'Markersize', 5); hold on;
        plot(y(:,1),y(:,2), 'b.', 'Markersize', 5);
        plot(z(:,1),z(:,2), 'g.', 'Markersize', 5);
        legend('x','y','z');
        title(sprintf("Distribution of x, y, and z at step %d (c = %d)",i,c));
        % print loss with fixed lambda and bw
        disp(sprintf('at %d step, LC = %9.5e, LF = %9.5e',i,LC,LF));
%         lambda

        if i<MAX_STEP
            input('Hit <return> to continue  '); 
        end
    end
    
    % gradient descent with current set of bw and lambda
%     if (i<=gamma)
        [LF,gradLF] = grad_LF(y,z,a,b);
        [LC,gradLC] = grad_LC(x,z);
        L = LC + lambda.*LF;
        gradL = gradLC + lambda.*gradLF;
        eta = 0.1;
%     end
    
    for k=1:INNER_STEP
        [z,eta,gradL,L] = grad_dc(x,y,z,a,b,lambda,eta,gradL,L,eta_tol);
    end
%     if (eta<eta_tol) % stop when no longer descent
%         break;
%     end

%     [z,eta_new,gradL] = grad_dc(x,y,z,a,b,lambda,eta,gradL);
%     z = znew; gradL = gradL_new;


    % update bw and lambda
%     c = c-dc; % decrese the multiplier for bw
%     a = c.*bw([z;y],1); % use rule of thumb to update bw
    if (i<gamma)%(a>afinal)
        a = a-da; % decrese a
%         c = c-dc; % decrese the multiplier for bw
%         a = c.*bw([z;y],1); % use rule of thumb to update bw
%         b-a
        b = a;
        
        lambda = lambda+dl; % increase lambda
    end
    
%     if (lambda<lambda_final)
%         lambda = lambda+dl; % increase lambda
%     end

%     if (lambda<10000) 
%         lambda = lambda*2;
%     end
    
    i = i+1;
end


% final L (loss)
disp(sprintf('at %d step, LC = %9.5e, LF = %9.5e',i,LC,LF));

% visualize results
% figure();

% plot in 1d
if (d==1)
    nbins=10;
    histogram(x,nbins,'FaceColor','r','Normalization','probability'); hold on;
    histogram(y,nbins,'FaceColor','b','Normalization','probability');
    histogram(z,nbins,'FaceColor','g','Normalization','probability');
    legend('x','y','z');
    title(sprintf("Distribution of x, y, and z at final step %d (c = %d)",i,c));
end

% plot in 2d
if (d==2)
    plot(x(:,1),x(:,2), 'r.', 'Markersize', 5); hold on;
    plot(y(:,1),y(:,2), 'b.', 'Markersize', 5);
    plot(z(:,1),z(:,2), 'g.', 'Markersize', 5);
    legend('x','y','z');
    title(sprintf("Distribution of x, y, and z at final step %d (c = %d)",i,c));
end

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

% grad_LC = @(x,z,N)(-1/N.*(x-z));

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
function [res1,res2] = grad_LF(y,z,a,b)

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
     tmp1(l,:,:) = (z(:,l)' - z(:,l))./a;
     tmp2(l,:,:) = (y(:,l)' - z(:,l))./b;
     tmp3(l,:,:) = (z(:,l)' - y(:,l))./a;
     tmp4(l,:,:) = (y(:,l)' - y(:,l))./b;
end

tmp5 = (sum(tmp1.*exp(-1/2.*sum(tmp1.^2,1)),3));
tmp6 = (sum(tmp2.*exp(-1/2.*sum(tmp2.^2,1)),3));

% normalizing constants
c1 = 1/(N^2)/((a*sqrt(2*pi))^d);
c2 = 1/(M*N)/((b*sqrt(2*pi))^d);
c3 = 1/(M*N)/((a*sqrt(2*pi))^d);
c4 = 1/(M^2)/((b*sqrt(2*pi))^d);

res2 = c1./(a).*(tmp5)' - c2./(b).*(tmp6)'; 

res1 = c1.*sum(exp(-1/2.*sum(tmp1.^2,1)),'all') - c2.*sum(exp(-1/2.*sum(tmp2.^2,1)),'all')...
    - c3.*sum(exp(-1/2.*sum(tmp3.^2,1)),'all') + c4.*sum(exp(-1/2.*sum(tmp4.^2,1)),'all');

% % approach 2 for grad: time-cosuming, but less memory? (not sure)

% res3 = zeros(N,d);
% tmp9 = 1/(N^2)/((a*sqrt(2*pi))^d);
% tmp10 = 1/(M*N)/((b*sqrt(2*pi))^d);
% for l=1:d
%     tmp5 = (z(:,l)' - z(:,l))./(a^2);
%     tmp6 = (y(:,l)' - z(:,l))./(b^2);
%     for i=1:N
%         tmp7 = sum(((z - z(i,:))./a).^2,2)';
%         tmp8 = sum(((y - z(i,:))./b).^2,2)';
% %         tmp5(i,:)
% %         exp(-1/2.*tmp7)
%         res3(i,l) = sum(tmp5(i,:).*exp(-1/2.*tmp7)).*tmp9...
%             - sum(tmp6(i,:).*exp(-1/2.*tmp8)).*tmp10;
%     end
%         
% end
% 
% res3


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
% res1 = -prod(1/(N-1)/((a*sqrt(2*pi))^d).*tmp3); % too small (prod = 0)
% res1 = -prod(1/(N-1).*tmp3);

% res3 = -N;
% for j=1:N
%     tmp1 = sum(((x(j,:)-x)./a).^2,2);
%     tmp2 = exp(-1/2.*tmp1);
%     
%     c = sum(tmp1.*tmp2,1); % numerator
%     d = sum(tmp2,1) - 1; % denominator
%     res3 = res3+c./d;
%     
% %     c = 0; % numerator
% %     d = 0; % denominator
% %     for i=1:N
% %         if (i~=j)
% %             c = c + tmp1(i)*tmp2(i);
% %             d = d + tmp2(i);
% %         end
% %     end
% %     res3 = res3+c/d;
% 
% end
% 
% res3 = res3./a

end

function [a] = bw(x,method)

N = length(x(:,1));
d = length(x(1,:));

% Rule of Thumb
a = (4/(d+2))^(1/(d+4))*(N^(-1/(d+4)))*mean(std(x));
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
%         P
%         Pnew
        if (anew>0 && Pnew < P)
            a = anew
            P = Pnew;
            grada = grada_new;
        end
        i = i+1;
%         norm(grada)
    end
%     i
%     Pnew
end

end

% % Rule of Thumb
% bw1 = @(x,N)(0.9*min(std(x),iqr(x)/1.34)*(N^(-1/5)));

%% Gradient Descent (one step)
function [znew,eta,gradL_new,Lnew] = grad_dc(x,y,z,a,b,lambda,eta,gradL,L,eta_tol)

% lambda: regularization parameter
% eta: learning rate
% gradL: gradient of the objective L wrt current z
% eta_tol: tolerance for (smallest) learning rate

eta = eta*2;
znew = z - eta.*gradL;
[LFnew,gradLF_new] = grad_LF(y,znew,a,b);
[LCnew,gradLC_new] = grad_LC(x,znew);
gradL_new = gradLC_new + lambda.*gradLF_new;
% [gradL_new] = grad_LC(x,znew) + lambda.*grad_LF(y,znew,a,b);
% n1 = norm(gradL);
% n2 = norm(gradL_new);

% k=0;
Lnew = LCnew + lambda.*LFnew;
while (Lnew > L && eta>eta_tol) %&& (abs(L-Lnew)>0.1))
% while ((n2 >= n1) && (abs(n1-n2)>0.1)) % find reasonable learning rate (when the gradient is closer to 0)
    eta = eta/2;
    znew = z - eta.*gradL;
    [LFnew,gradLF_new] = grad_LF(y,znew,a,b);
    [LCnew,gradLC_new] = grad_LC(x,znew);
    Lnew = LCnew + lambda.*LFnew;
    gradL_new = gradLC_new + lambda.*gradLF_new;
%     gradL_new = grad_LC(x,znew) + lambda.*grad_LF(y,znew,a,b);
%     n2 = norm(gradL_new);
%     k=k+1;
%     if (k>=2)
%         k
%     end
end

% gradL_new = gradLC_new + lambda.*gradLF_new;
if (Lnew > L)
    znew = z;
    Lnew = L;
    gradL_new = gradL;
end

end

