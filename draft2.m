clear;

N=500;
M=500;
% x = [(0:10)' (10:-1:0)']; % samples drawn from rho (N-by-d matrix)
% x = [1 1; 2 2; 3 3];
x = mvnrnd([1;1],[10 0;0 1],N);
theta = pi/3;
y = [cos(theta).*x(:,1)+sin(theta).*x(:,2), -sin(theta).*x(:,1)+cos(theta).*x(:,2)];
y = y+1;
% y = mvnrnd([20;20],[10 0;0 1],M); % samples drawn from mu (M-by-d matrix)

[res1,res2] = grad_LR_w(x,y);
% res1
% norm(res2)

% [res3,res4] = grad_LC_w(x,y);
% res3
% norm(res4)

% gamma = 5e3;
% gradL = res4 + gamma.*res2;
% norm(gradL);
% 
% eta=0.1;
% xnew = x - eta.*gradL;

% 
% [res1,res2] = grad_LC_w(x,y);
% res1
% norm(res2)

% figure();
% plot(x(:,1),x(:,2), 'r.', 'Markersize', 5); hold on;
% plot(y(:,1),y(:,2), 'b.', 'Markersize', 5);
% plot(xnew(:,1),xnew(:,2), 'g.', 'Markersize', 5);


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
% eps = 1e-32; % some small number
eps_sqr = 1e-3;%eps^2;

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

% approach 2 for grad: much slower (iteration)
res3 = 0;
for i=1:N
    for j=1:N
        res3 = res3 + ...
            ((norm(w(i,:)-w(j,:))^2)/(norm(x(i,:)-x(j,:))^2 + eps^2)-1)^2;
    end
end
res3 = res3/(N^2)/2;

res4 = zeros(N,d);

for i=1:N
    for j=1:N
%         if (i~=j)
            res4(i,:) = res4(i,:) + ((norm(w(i,:)-w(j,:))^2)/(norm(x(i,:)-x(j,:))^2 + eps^2)-1)...
                .*(w(i,:)-w(j,:))./(norm(x(i,:)-x(j,:))^2 + eps^2);
%         end
    end
end

res4 = res4.*4./(N^2);

res1
res3
'\n'
res2
res4
norm(res2-res4)

% res1
% res3

end
