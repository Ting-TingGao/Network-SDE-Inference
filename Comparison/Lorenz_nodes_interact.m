%stochastic Lorenz system simulation
% for paper "Inferring stochastic dynamics of complex systems", T.-T. Gao,
% B. Barzel, and G. Yan

% equation from paper 'Learning stochastic dynamical systems with neural
% networks mimicking the euler-maruyama scheme'
% and 'Dynamics and transitions of the coupled Lorenz system'

% % % Comparison dataset
% 
clc, clf, clear, close all
tic

dim=3;
dt = 0.01;
tspan = dt:dt:100;

% A = [0,0,0,0,1;
%     1,0,0,0,0;
%     0,1,0,0,0;
%     0,0,1,0,0;
%     0,0,0,1,0];


E_list = textread('20nodes.txt');
E_list = E_list(3:length(E_list),:);
E_num = length(E_list);
E_list = E_list+1;
A = zeros(20);
for i = 1:E_num
    A(E_list(i,2),E_list(i,1))=1;
end

n1 = length(A);
n = n1*dim;

T = length(tspan);

x0 = rand(1,n);

s = normrnd(0,sqrt(dt),T,n);
x = [];
x_determ = [];
x(1,:) = x0;
x_determ(1,:) = x0;
gamma = 1; % gamma lower than 0.02 the system will break down.
epsilon = 1;

% generate the stochastic trajectories and save as x
for t = 1:T
    for i = 1:n1
        tmp = 0;
        for j = 1:n1
            tmp = tmp+A(i,j)*x(t,3*j-2);
        end
        x(t+1,3*i-2) = x(t,3*i-2)+(10*x(t,3*i-1)-(10+2/gamma)*x(t,3*i-2)+epsilon*tmp)*dt+x(t,3*i-2)/sqrt(gamma)*s(t,3*i-2);
        x(t+1,3*i-1) = x(t,3*i-1)+((28-x(t,3*i))*x(t,3*i-2)-(1+2/gamma)*x(t,3*i-1))*dt+(28-x(t,3*i))/sqrt(gamma)*s(t,3*i-1);
        x(t+1,3*i) = x(t,3*i)+(x(t,3*i-2)*x(t,3*i-1)-((8/3)+4/gamma)*x(t,3*i))*dt+x(t,3*i-1)/sqrt(gamma)*s(t,3*i);
        F(t+1,3*i-2) = (10*x(t,3*i-1)-(10+2/gamma)*x(t,3*i-2)+epsilon*tmp);
        F(t+1,3*i-1) = (28-x(t,3*i))*x(t,3*i-2)-(1+2/gamma)*x(t,3*i-1);
        F(t+1,3*i) = x(t,3*i-2)*x(t,3*i-1)-((8/3)+4/gamma)*x(t,3*i);
        sto(t+1,3*i-2) = x(t,3*i-2)/sqrt(gamma);
        sto(t+1,3*i-1) = (28-x(t,3*i))/sqrt(gamma);
        sto(t+1,3*i) = x(t,3*i-1)/sqrt(gamma);
    end
end


% generate the deterministic trajectories and save as x_determ, 
% F is force field
for t = 1:T
    for i = 1:n1
        tmp = 0;
        for j = 1:n1
            tmp = tmp+A(i,j)*x_determ(t,3*j-2);
        end
        x_determ(t+1,3*i-2) = x_determ(t,3*i-2)+(10*x_determ(t,3*i-1)-(10+2/gamma)*x_determ(t,3*i-2)+epsilon*tmp)*dt;
        x_determ(t+1,3*i-1) = x_determ(t,3*i-1)+((28-x_determ(t,3*i))*x_determ(t,3*i-2)-(1+2/gamma)*x_determ(t,3*i-1))*dt;
        x_determ(t+1,3*i) = x_determ(t,3*i)+(x_determ(t,3*i-2)*x_determ(t,3*i-1)-((8/3)+4/gamma)*x_determ(t,3*i))*dt;
    end
end


N = length(x(:,1));
t = (1:1:N);
figure(1)
subplot(1,2,1)
colormap(parula)
patch(x(:,1),x(:,2),x(:,3),t,'FaceColor','None','EdgeColor','interp')
colorbar
view(3)
subplot(1,2,2)
colormap(parula)
patch(x_determ(:,1),x_determ(:,2),x_determ(:,3),t,'FaceColor','None','EdgeColor','interp','linewidth',2)
colorbar
view(3)

figure(2)
colormap(parula)
patch(x(:,1),x(:,2),x(:,3),t,'FaceColor','None','EdgeColor','interp')
colorbar
view(3)

%csvwrite('Lorenztspan_gamma1.csv',tspan);
%add noise
STD = zeros(1,n);
noise = zeros(length(x),n);
noise_p = 8;
for i = 1:n
    STD(i) = std(x(:,i));
    noise(:,i) = randn(length(x),1).*STD(i)*noise_p*0.01;
end

x = x+noise;

% derivative
dxdt = (x(2:end,:)-x(1:end-1,:))./dt;
xt1 = x(2:end,:);
x = x(1:end-1,:);
x_dterm = x_determ(1:end-1,:);
csvwrite('Lorenz_x_stochastic_gamma1_noise_8.csv',x);
%csvwrite('Lorenz_dx_stochastic_gamma1_noise.csv',dxdt);
%csvwrite('Lorenz2nodesAdj.csv', A);
%csvwrite('Lorenz_x_determ_gamma1_noise.csv',x_determ);

F = F(1:end-1,:);
sto = sto(1:end-1,:);
%% sub library
xdata = zeros(length(xt1)*n1,6);
x1data = zeros(length(x)*n1,6);
dxdata = zeros(length(dxdt)*n1,3);
Fdata = zeros(length(F)*n1,3);
stodata = zeros(length(sto)*n1,3);
for i = 1:n1
    xdata((i-1)*length(x)+1:i*length(x),1:3) = x(:,3*i-2:3*i);
    x1data((i-1)*length(xt1)+1:i*length(xt1),1:3) = xt1(:,3*i-2:3*i);
    dxdata((i-1)*length(dxdt)+1:i*length(dxdt),1:3) = dxdt(:,3*i-2:3*i);
    Fdata((i-1)*length(F)+1:i*length(F),1:3) = F(:,3*i-2:3*i);
    stodata((i-1)*length(sto)+1:i*length(sto),1:3) = sto(:,3*i-2:3*i);
end
for t = 1:length(x)
    for i = 1:n1
        tmp1 = 0;
        tmp2 = 0;
        tmp3 = 0;
        for j = 1:n1
            tmp1 = tmp1+A(i,j)*x(t,3*j-2);%xj
            tmp2 = tmp2+A(i,j)*(x(t,3*j-2)*x(t,3*i-2));%xixj
            tmp3 = tmp3+A(i,j)*(x(t,3*j-2)-x(t,3*i-2));%xj-xi
        end
        xdata((i-1)*length(x)+t,4) = tmp1;
        xdata((i-1)*length(x)+t,5) = tmp2;
        xdata((i-1)*length(x)+t,6) = tmp3;
    end
            
end
% csvwrite('Lorenz_x_stochastic_gamma075_withlib.csv',xdata);
% csvwrite('Lorenz_xt1_stochastic_gamma1_withlib.csv',x1data);
% csvwrite('Lorenz_dxdt_stochastic_gamma1_all.csv',dxdata);
% csvwrite('Lorenz_F_stochastic_gamma1_all.csv',Fdata);
% csvwrite('Lorenz_sto_stochastic_gamma1_all.csv',stodata);
toc
%% single Lorenz node
% clc, clf, clear, close all
% tic
% dim=3;
% dt = 0.01;
% tspan = dt:dt:500;
% T = length(tspan);
% 
% x0 = rand(1,3);
% 
% s = normrnd(0,sqrt(dt),T,3);
% x = [];
% x_determ = [];
% x(1,:) = x0;
% x_determ(1,:) = x0;
% gamma = 10; % gamma lower than 0.02 the system will break down.
% 
% 
% % generate the stochastic trajectories and save as x
% for t = 1:T 
%     x(t+1,1) = x(t,1)+(10*x(t,2)-(10+2/gamma)*x(t,1))*dt+x(t,1)/sqrt(gamma)*s(t,1);
%     x(t+1,2) = x(t,2)+((28-x(t,3))*x(t,1)-(1+2/gamma)*x(t,2))*dt+(28-x(t,3))/sqrt(gamma)*s(t,2);
%     x(t+1,3) = x(t,3)+(x(t,1)*x(t,2)-((8/3)+4/gamma)*x(t,3))*dt+x(t,2)/sqrt(gamma)*s(t,3);
%     F(t+1,1) = 10*x(t,2)-(10+2/gamma)*x(t,1);
%     F(t+1,2) = (28-x(t,3))*x(t,1)-(1+2/gamma)*x(t,2);
%     F(t+1,3) = x(t,1)*x(t,2)-((8/3)+4/gamma)*x(t,3);
%     S(t+1,1) = x(t,1)/sqrt(gamma);
%     S(t+1,2) = (28-x(t,3))/sqrt(gamma);
%     S(t+1,3) = x(t,2)/sqrt(gamma);
% end
% 
% % for t = 1:T 
% %     x(t+1,1) = x(t,1)+(10*x(t,2)-(10+2/gamma)*x(t,1))*dt+x(t,1)/sqrt(gamma)*s(t,1);
% %     x(t+1,2) = x(t,2)+((28-x(t,3))*x(t,1)-(1+2/gamma)*x(t,2))*dt+x(t,2)/sqrt(gamma)*s(t,2);
% %     x(t+1,3) = x(t,3)+(x(t,1)*x(t,2)-((8/3)+4/gamma)*x(t,3))*dt+x(t,3)/sqrt(gamma)*s(t,3);
% %     F(t+1,1) = 10*x(t,2)-(10+2/gamma)*x(t,1);
% %     F(t+1,2) = (28-x(t,3))*x(t,1)-(1+2/gamma)*x(t,2);
% %     F(t+1,3) = x(t,1)*x(t,2)-((8/3)+4/gamma)*x(t,3);
% %     S(t+1,1) = x(t,1)/sqrt(gamma);
% %     S(t+1,2) = x(t,2)/sqrt(gamma);
% %     S(t+1,3) = x(t,3)/sqrt(gamma);
% % end
% 
% 
% % generate the deterministic trajectories and save as x_determ, 
% for t = 1:T
%     x_determ(t+1,1) = x_determ(t,1)+(10*x_determ(t,2)-(10+2/gamma)*x_determ(t,1))*dt;
%     x_determ(t+1,2) = x_determ(t,2)+((28-x_determ(t,3))*x_determ(t,1)-(1+2/gamma)*x_determ(t,2))*dt;
%     x_determ(t+1,3) = x_determ(t,3)+(x_determ(t,1)*x_determ(t,2)-((8/3)+4/gamma)*x_determ(t,3))*dt;
% end
% N = length(x(:,1));
% t = (1:1:N);
% figure(1)
% subplot(1,2,1)
% colormap(parula)
% patch(x(:,1),x(:,2),x(:,3),t,'FaceColor','None','EdgeColor','interp')
% colorbar
% view(3)
% subplot(1,2,2)
% colormap(parula)
% patch(x_determ(:,1),x_determ(:,2),x_determ(:,3),t,'FaceColor','None','EdgeColor','interp','linewidth',2)
% colorbar
% view(3)
% 
% figure(2)
% colormap(parula)
% patch(x(:,1),x(:,2),x(:,3),t,'FaceColor','None','EdgeColor','interp')
% colorbar
% view(3)
% % derivative
% dxdt = (x(2:end,:)-x(1:end-1,:))./dt;
% dx1dt =(x_determ(2:end,:)-x_determ(1:end-1,:))./dt;
% xt1 = x(2:end,:);
% x = x(1:end-1,:);
% x_determ = x_determ(1:end-1,:);
% 
% csvwrite('Lorenzsingle_x_stochastic_gamma1.csv',x);
% csvwrite('Lorenzsingle_dx_stochastic_gamma1.csv',dxdt);
% csvwrite('Lorenzsingle_x_gamma1.csv',x_determ);
% csvwrite('Lorenzsingle_dx_gamma1.csv',dx1dt);
% csvwrite('Lorenzsingle_F_gamma1.csv',F);
% csvwrite('Lorenzsingle_S_gamma1.csv',S);
% csvwrite('Lorenzsingle_xt1_gamma1.csv',xt1);
% 
% % A = [0,1;
% %     1,0;];
% % Xall = [x,x];
% %csvwrite('Lorenzsingle_x_lagna_spurious_adj.csv',A);
% %csvwrite('Lorenzsingle_x_determ_gamma10_lagna.csv',Xall);
% toc

%% diagnal noise
% 
% clc, clf, clear, close all
% tic
% 
% dim=3;
% dt = 0.01;
% tspan = dt:dt:100;
% 
% % A = [0,0,0,0,1;
% %     1,0,0,0,0;
% %     0,1,0,0,0;
% %     0,0,1,0,0;
% %     0,0,0,1,0];
% 
% 
% E_list = textread('20nodes.txt');
% E_list = E_list(3:length(E_list),:);
% E_num = length(E_list);
% E_list = E_list+1;
% A = zeros(20);
% for i = 1:E_num
%     A(E_list(i,2),E_list(i,1))=1;
% end
% 
% n1 = length(A);
% n = n1*dim;
% 
% T = length(tspan);
% 
% x0 = rand(1,n);
% 
% s = normrnd(0,sqrt(dt),T,n);
% x = [];
% x_determ = [];
% x(1,:) = x0;
% x_determ(1,:) = x0;
% gamma = 0.25; % gamma lower than 0.02 the system will break down.
% epsilon = 1;
% 
% % generate the stochastic trajectories and save as x
% for t = 1:T
%     for i = 1:n1
%         tmp = 0;
%         for j = 1:n1
%             tmp = tmp+A(i,j)*x(t,3*j-2);
%         end
%         x(t+1,3*i-2) = x(t,3*i-2)+(10*x(t,3*i-1)-(10+2/gamma)*x(t,3*i-2)+epsilon*tmp)*dt+x(t,3*i-2)/sqrt(gamma)*s(t,3*i-2);
%         x(t+1,3*i-1) = x(t,3*i-1)+((28-x(t,3*i))*x(t,3*i-2)-(1+2/gamma)*x(t,3*i-1))*dt+x(t,3*i-1)/sqrt(gamma)*s(t,3*i-1);
%         x(t+1,3*i) = x(t,3*i)+(x(t,3*i-2)*x(t,3*i-1)-((8/3)+4/gamma)*x(t,3*i))*dt+x(t,3*i)/sqrt(gamma)*s(t,3*i);
%         F(t+1,3*i-2) = (10*x(t,3*i-1)-(10+2/gamma)*x(t,3*i-2)+epsilon*tmp);
%         F(t+1,3*i-1) = (28-x(t,3*i))*x(t,3*i-2)-(1+2/gamma)*x(t,3*i-1);
%         F(t+1,3*i) = x(t,3*i-2)*x(t,3*i-1)-((8/3)+4/gamma)*x(t,3*i);
%         sto(t+1,3*i-2) = x(t,3*i-2)/sqrt(gamma);
%         sto(t+1,3*i-1) = x(t,3*i-1)/sqrt(gamma);
%         sto(t+1,3*i) = x(t,3*i)/sqrt(gamma);
%     end
% end
% 
% 
% % generate the deterministic trajectories and save as x_determ, 
% % F is force field
% for t = 1:T
%     for i = 1:n1
%         tmp = 0;
%         for j = 1:n1
%             tmp = tmp+A(i,j)*x_determ(t,3*j-2);
%         end
%         x_determ(t+1,3*i-2) = x_determ(t,3*i-2)+(10*x_determ(t,3*i-1)-(10+2/gamma)*x_determ(t,3*i-2)+epsilon*tmp)*dt;
%         x_determ(t+1,3*i-1) = x_determ(t,3*i-1)+((28-x_determ(t,3*i))*x_determ(t,3*i-2)-(1+2/gamma)*x_determ(t,3*i-1))*dt;
%         x_determ(t+1,3*i) = x_determ(t,3*i)+(x_determ(t,3*i-2)*x_determ(t,3*i-1)-((8/3)+4/gamma)*x_determ(t,3*i))*dt;
%     end
% end
% 
% 
% N = length(x(:,1));
% t = (1:1:N);
% figure(1)
% subplot(1,2,1)
% colormap(parula)
% patch(x(:,1),x(:,2),x(:,3),t,'FaceColor','None','EdgeColor','interp')
% colorbar
% view(3)
% subplot(1,2,2)
% colormap(parula)
% patch(x_determ(:,1),x_determ(:,2),x_determ(:,3),t,'FaceColor','None','EdgeColor','interp','linewidth',2)
% colorbar
% view(3)
% 
% figure(2)
% colormap(parula)
% patch(x(:,1),x(:,2),x(:,3),t,'FaceColor','None','EdgeColor','interp')
% colorbar
% view(3)
% 
% %csvwrite('Lorenztspan_gamma1.csv',tspan);
% 
% % derivative
% dxdt = (x(2:end,:)-x(1:end-1,:))./dt;
% xto = x(2:end,:); %for SDEnet use
% x = x(1:end-1,:);
% x_dterm = x_determ(1:end-1,:);
% csvwrite('Lorenz_x_diag_stochastic_gamma025.csv',x);
% csvwrite('Lorenz_dx_diag_stochastic_gamma025.csv',dxdt);
% %csvwrite('Lorenz2nodesAdj.csv', A);
% csvwrite('Lorenz_x_diag_determ_gamma025.csv',x_determ);
% 
% F = F(1:end-1,:);
% sto = sto(1:end-1,:);
% %% sub library
% x0data = zeros(length(x)*n1,6);
% xdata = zeros(length(x)*n1,6);
% dxdata = zeros(length(dxdt)*n1,3);
% Fdata = zeros(length(F)*n1,3);
% stodata = zeros(length(sto)*n1,3);
% for i = 1:n1
%     x0data((i-1)*length(xto)+1:i*length(xto),1:3) = xto(:,3*i-2:3*i);
%     xdata((i-1)*length(x)+1:i*length(x),1:3) = x(:,3*i-2:3*i);
%     dxdata((i-1)*length(dxdt)+1:i*length(dxdt),1:3) = dxdt(:,3*i-2:3*i);
%     Fdata((i-1)*length(F)+1:i*length(F),1:3) = F(:,3*i-2:3*i);
%     stodata((i-1)*length(sto)+1:i*length(sto),1:3) = sto(:,3*i-2:3*i);
% end
% for t = 1:length(x)
%     for i = 1:n1
%         tmp1 = 0;
%         tmp2 = 0;
%         tmp3 = 0;
%         for j = 1:n1
%             tmp1 = tmp1+A(i,j)*x(t,3*j-2);%xj
%             tmp2 = tmp2+A(i,j)*(x(t,3*j-2)*x(t,3*i-2));%xixj
%             tmp3 = tmp3+A(i,j)*(x(t,3*j-2)-x(t,3*i-2));%xj-xi
%         end
%         xdata((i-1)*length(x)+t,4) = tmp1;
%         xdata((i-1)*length(x)+t,5) = tmp2;
%         xdata((i-1)*length(x)+t,6) = tmp3;
%     end
%             
% end
% csvwrite('Lorenz_x_diag_stochastic_gamma025_withlib.csv',xdata);
% csvwrite('Lorenz_xto_diag_stochastic_gamma025_withlib.csv',x0data);
% csvwrite('Lorenz_dxdt_diag_stochastic_gamma025_all.csv',dxdata);
% csvwrite('Lorenz_F_diag_stochastic_gamma025_all.csv',Fdata);
% csvwrite('Lorenz_sto_diag_stochastic_gamma025_all.csv',stodata);
% toc