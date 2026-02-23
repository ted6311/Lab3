%% Part 3.1
% Compare Newton and Gauss-Newton

%% a)
clear;
m = 1000; s = 0.25; alpha = 4; beta = -0.25; gamma = 4;
xp = 10*rand(m,1);
J = zeros(m,3);
J(:,1) = exp(beta*xp).*sin(gamma*xp);
J(:,2) = alpha*xp.*exp(beta*xp).*sin(gamma*xp);
J(:,3) = alpha*xp.*exp(beta*xp).*cos(gamma*xp);
v = randn(m-3,1); v = (sqrt(m)*s/norm(v))*v;
yp = alpha*exp(beta*xp).*sin(gamma*xp) + null(J')*v;


A = alpha;
B = linspace(beta-1, beta+1, 100);
C = linspace(gamma-1, gamma+1, 100);

[Bgrid, Cgrid] = meshgrid(B,C);
F = zeros(size(Bgrid));

for i = 1: length(B)
    for j = 1:length(C)
        ymodel = A*exp(Bgrid(i,j)*xp).*sin(Cgrid(i,j)*xp);
        r = ymodel - yp;
        F(i,j) = 0.5*sum(r.^2);
    end
end

figure;
contourf(Bgrid, Cgrid, F, 50)
hold on
plot(beta, gamma, 'r*', 'MarkerSize', 12)
colorbar
xlabel('B')
ylabel('C')
title('Objective function f(B,C)')


%% b) 

function r = residual(u, xp, yp)
    A = u(1);
    B = u(2);
    C = u(3);
    ymodel = A*exp(B*xp).*sin(C*xp);
    r = ymodel - yp;
end

function J = jacobian(u, xp)
    A = u(1);
    B = u(2);
    C = u(3);
    m = length(xp);
    J = zeros(m,3);

    J(:,1) = exp(B*xp).*sin(C*xp);
    J(:,2) = A*xp.*exp(B*xp).*sin(C*xp);
    J(:,3) = A*xp.*exp(B*xp).*cos(C*xp);

end

%initial guess
u_newton = [3; -0.1; 3];
u_gn     = [3; -0.1; 3];


% Newtons metod
   
j = 50;
tolerans = 1e-8;

for k = 1:j
    
    r = residual(u_newton, xp, yp);
    J = jacobian(u_newton, xp);
    
    g = J'*r;   % gradient
    
    % Hessian
    H = J'*J;
    
    A = u_newton(1);
    B = u_newton(2);
    C = u_newton(3);
    
    for i = 1:length(xp)
        
        x = xp(i);
        
        % Second derivatives
        d2AA = 0;
        d2BB = A*x^2*exp(B*x)*sin(C*x);
        d2CC = -A*x^2*exp(B*x)*sin(C*x);
        d2BC = A*x^2*exp(B*x)*cos(C*x);
        
        H(2,2) = H(2,2) + r(i)*d2BB;
        H(3,3) = H(3,3) + r(i)*d2CC;
        H(2,3) = H(2,3) + r(i)*d2BC;
        H(3,2) = H(2,3);
        
    end
    
    lambda = 1e-3;                 
    delta = -(H + lambda*eye(3))\g;
    u_newton = u_newton + delta;
    
    if norm(delta) < tolerans
        break
    end
    
end



% Gauss-Newton



for k = 1:j
    
    r = residual(u_gn, xp, yp);
    J = jacobian(u_gn, xp);
    
    % QR-dekomposition
    delta = -J\r;
    
    u_gn = u_gn + delta;
    
    if norm(delta) < tolerans
        break
    end
    
end


%Plot
u_opt = u_gn;

xplot = linspace(0,10,500)';

y_init = 3*exp(-0.1*xplot).*sin(3*xplot);
y_opt = u_opt(1)*exp(u_opt(2)*xplot).*sin(u_opt(3)*xplot);

figure
scatter(xp, yp, 10, 'filled')
hold on
plot(xplot, y_init, 'r', 'LineWidth',2)
plot(xplot, y_opt, 'k', 'LineWidth',2)
legend('Data','Initial guess','Optimized model')