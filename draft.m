clear;

a(1) = 3;

max_steps = 1000; % maximum number of steps
eta(1) = 1e-1; % initial learning rate

[P,grada] = grad_tmp(a(1));

tol = 1e-32; % (relative) tolerance
i = 0;
while (i<max_steps && norm(grada)>tol)
    eta(i+2) = eta(i+1)*2;
%     anew = a(i+1) - eta(i+2).*grada;
    anew = a(i+1) - eta(i+2)/(1+2*eta(i+2)).*grada;
    [Pnew,grada_new] = grad_tmp(anew);
    k=0;
%     P
%     Pnew
    while (Pnew > P) %&& eta > 1e-32)%Pnew > P) %&& (abs(L-Lnew)>0.1))
        eta(i+2) = eta(i+2)/2;
%         anew = a(i+1) - eta(i+2).*grada;
        anew = a(i+1) - eta(i+2)/(1+2*eta(i+2)).*grada;
        [Pnew,grada_new] = grad_tmp(anew);
        k=k+1;
%         P
%         Pnew
    end
    
    if (Pnew < P)
        a(i+2) = anew;
        P = Pnew;
        grada = grada_new;
%         eta(i+2)
%         eta(i+2)/(1+2*eta(i+2))
    end
%     a(i+2)
%     input('Hit <return> to continue  ');
    
    i = i+1;
%     norm(grada)
end

figure();
plot(0:i,eta,'b.', 'Markersize', 10);
xlabel('number of steps');
ylabel('\eta');
title('learning rates');

figure();
plot(0:i,a,'r.', 'Markersize', 10);
xlabel('number of steps');
ylabel('a');
title('grad dc');

function [res1,res2] = grad_tmp(a)

res1 = a.^2;

res2 = 2.*a;

end