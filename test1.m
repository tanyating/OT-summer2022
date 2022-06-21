clear;


%% define distributions (samples)

N = 500; % number of samples of x
M = 500; % number of samples of y

% d = 1; % dimension
% x = mvnrnd(0,1,N); % samples drawn from rho (N-by-d matrix)
% y = mvnrnd(5,1,M); % samples drawn from mu (M-by-d matrix)

d = 2; % dimension
x = mvnrnd([1;1],eye(2),N); % samples drawn from rho (N-by-d matrix)
y = mvnrnd([10;5],eye(2),M); % samples drawn from mu (M-by-d matrix)

z = x; % start the transport with the original samples
lambda = 1000; % regularization parameter (large)
eta = 0.1; % learning rate
MAX_STEP = 8000; % maximum steps of grad dc
for i=1:MAX_STEP
    z = grad_dc(x,y,z,lambda,eta);
end

% visualize results
figure();

% % plot in 1d
% nbins=10;
% histogram(x,nbins,'Normalization','probability'); hold on;
% histogram(y,nbins,'Normalization','probability');
% histogram(z,nbins,'Normalization','probability');
% legend('x','y','z');

% plot in 2d
plot(x(:,1),x(:,2), '.', 'Markersize', 5); hold on;
plot(y(:,1),y(:,2), '.', 'Markersize', 5);
plot(z(:,1),z(:,2), '.', 'Markersize', 5);
legend('x','y','z');

%---------------------------------------------

%% Gradient 

% gradient for L_C
function [res] = grad_LC(x,z)

N = length(x(:,1));
res = -1/N.*(x-z);

end

% grad_LC = @(x,z,N)(-1/N.*(x-z));

% gradient for L_F (without lambda)
function [res] = grad_LF(y,z,a,b)

N = length(z(:,1));
M = length(y(:,1));
d = length(z(1,:));
res = zeros(N,d);
for i=1:N
    res(i,:) = sum(1./(a.^3)./(N^2).*(z-z(i,:)).*exp(-1/2.*((z-z(i,:))./a).^2)...
    - 1./(b.^3)./(M*N).*(y-z(i,:)).*exp(-1/2.*((y-z(i,:))./b).^2),1);
%     - 1/(a^3)/(M*N).*(y-z(i)).*exp(-1/2.*((y-z(i))./a).^2));
end
res = res./(sqrt(2*pi));

end

%% KDE bandwidths (default Gaussian kernel)

% deep method gradient wrt a (max log-likelihood)
function [res] = grad_bw(x,a)

% x: N samples (data)
% a: bandwidth

N = length(x(:,1));
res = -N;

for j=1:N
    tmp1 = ((x(j,:)-x)./a).^2;
    tmp2 = exp(-1/2.*tmp1);
    c = sum(tmp1.*tmp2,1); % numerator
    d = sum(tmp2,1) - 1; % denominator
    res = res+c./d;
%     c = 0; % numerator
%     d = 0; % denominator
%     for i=1:N
%         if (i~=j)
%             c = c + tmp1(i)*tmp2(i);
%             d = d + tmp2(i);
%         end
%     end
%     res = res+c/d;
end

res = res./a;

end

function [a] = bw(x,method)

N = length(x);

% Rule of Thumb
a = 0.9.*min(std(x),iqr(x)/1.34).*(N^(-1/5));
if (method == 1)
    return;
    
elseif (method == 2)
    % deep method (gradient ascent)
%     a = 0.5; % initial guess
    max_steps = 10; % maximum number of steps
    eta = 1e-3; % learning rate (has to be very small?)
    for i=1:max_steps
        a = a + eta.*grad_bw(x,a);
    end
end

end

% % Rule of Thumb
% bw1 = @(x,N)(0.9*min(std(x),iqr(x)/1.34)*(N^(-1/5)));

%% Gradient Descent (one step)
function [z] = grad_dc(x,y,z,lambda,eta)

% lambda: regularization parameter
% eta: learning rate

% a = bw(z,1); % bandwidth for rho_T
% b = bw(y,1); % bandwidth for mu
a = bw([z;y],1); % use a common, large bandwidth for rho_T and mu
b = a;
z = z - eta.*(grad_LC(x,z) + lambda.*grad_LF(y,z,a,b));

end

