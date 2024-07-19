%stochastic Lorenz system simulation
% for paper "Inferring stochastic dynamics of complex systems", T.-T. Gao,
% B. Barzel, and G. Yan

% equation from paper 'Learning stochastic dynamical systems with neural
% networks mimicking the euler-maruyama scheme'
% and 'Dynamics and transitions of the coupled Lorenz system'

clc, clf, clear, close all
tic

dim=3;
dt = 0.01;
tspan = dt:dt:100;

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

x0 = csvread('Lorenz_stochastic_005_inital_20.csv');%rand(1,n);

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
        x(t+1,3*i-2) = x(t,3*i-2)+(10*x(t,3*i-1)-(10+2/gamma)*x(t,3*i-2)+epsilon*tmp)*dt+abs(x(t,3*i-2)/sqrt(gamma))*s(t,3*i-2);
        x(t+1,3*i-1) = x(t,3*i-1)+((28-x(t,3*i))*x(t,3*i-2)-(1+2/gamma)*x(t,3*i-1))*dt+abs((28-x(t,3*i))/sqrt(gamma))*s(t,3*i-1);
        x(t+1,3*i) = x(t,3*i)+(x(t,3*i-2)*x(t,3*i-1)-((8/3)+4/gamma)*x(t,3*i))*dt+abs(x(t,3*i-1)/sqrt(gamma))*s(t,3*i);
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
        F(t+1,3*i-2) = (10*x_determ(t,3*i-1)-(10+2/gamma)*x_determ(t,3*i-2)+epsilon*tmp);
        F(t+1,3*i-1) = (28-x_determ(t,3*i))*x_determ(t,3*i-2)-(1+2/gamma)*x_determ(t,3*i-1);
        F(t+1,3*i) = x_determ(t,3*i-2)*x_determ(t,3*i-1)-((8/3)+4/gamma)*x_determ(t,3*i);
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

%csvwrite('Lorenz_stochastic_in005_200.csv',x);
%csvwrite('Lorenz_determ_in005_200.csv',x_determ);
% writematrix(x,'Lorenz_stochastic_in005_50.csv');
% writematrix(x_determ,'Lorenz_determ_in005_50.csv');
toc


%% delete links
% b1 = (A==1);
% Num1 = sum(b1(:));
% s1 = 0;
% adj = A;
% while s1<Num1*0.02
% I = randperm(length(A),1);
% J = randperm(length(A),1);
% if adj(I,J) ~= 0
%     adj(I,J) = 0;
%     s1 = s1+1;
% end
% end
% b2 = (adj==1);
% Num2 = sum(b2(:));
% 
% csvwrite('missing2%_20_adj.csv',adj);
%% inferred with diffusion
x_i = [];
x_i(1,:) = x0;
for t = 1:T
    for i = 1:n1
        tmp = 0;
        for j = 1:n1
            tmp = tmp+A(i,j)*x_i(t,3*j-2);
        end
        x_i(t+1,3*i-2) = x_i(t,3*i-2)+(10.07386531*x_i(t,3*i-1)-12.12511225*x_i(t,3*i-2)+1.0015744*tmp)*dt+abs(1.01*x(t,3*i-2)/sqrt(gamma))*s(t,3*i-2);
        x_i(t+1,3*i-1) = x_i(t,3*i-1)+((27.60717738-0.98592823*x_i(t,3*i))*x_i(t,3*i-2)-2.95479691*x_i(t,3*i-1))*dt+abs((30.5-1.09*x(t,3*i))/sqrt(gamma))*s(t,3*i-1);
        x_i(t+1,3*i) = x_i(t,3*i)+(1.00233568*x_i(t,3*i-2)*x_i(t,3*i-1)-6.67988203*x_i(t,3*i))*dt+abs(1.01*x(t,3*i-1)/sqrt(gamma))*s(t,3*i);
    end
end

% Comparison of force field
t = (1:1:N);
figure(3)
colormap(parula)
patch(x_i(:,1),x_i(:,2),x_i(:,3),t,'FaceColor','None','EdgeColor','interp')
colorbar
view(3)

%% Inferred without diffusion
x_i = [];
x_i(1,:) = x0;
for t = 1:T
    for i = 1:n1
        tmp = 0;
        for j = 1:n1
            tmp = tmp+A(i,j)*x_i(t,3*j-2);
        end
        x_i(t+1,3*i-2) = x_i(t,3*i-2)+(10.07386531*x_i(t,3*i-1)-12.12511225*x_i(t,3*i-2)+1.0015744*tmp)*dt;
        x_i(t+1,3*i-1) = x_i(t,3*i-1)+((27.60717738-0.98592823*x_i(t,3*i))*x_i(t,3*i-2)-2.95479691*x_i(t,3*i-1))*dt;
        x_i(t+1,3*i) = x_i(t,3*i)+(1.00233568*x_i(t,3*i-2)*x_i(t,3*i-1)-6.67988203*x_i(t,3*i))*dt;
        Fi(t+1,3*i-2) = 10.07386531*x_i(t,3*i-1)-12.12511225*x_i(t,3*i-2)+1.0015744*tmp;
        Fi(t+1,3*i-1) = (27.60717738-0.98592823*x_i(t,3*i))*x_i(t,3*i-2)-2.95479691*x_i(t,3*i-1);
        Fi(t+1,3*i) = 1.00233568*x_i(t,3*i-2)*x_i(t,3*i-1)-6.67988203*x_i(t,3*i);
    end
end

% Comparison of force field
figure;
for i = 1:20
h = quiver(x_determ(10,3*i-2),x_determ(10,3*i-1),F(10,3*i-2),F(10,3*i-1),0.02,'k')
set(h,'MaxHeadSize',0,'linewidth',0.8) 
hold on
hi = quiver(x_i(10,3*i-2),x_i(10,3*i-1),Fi(10,3*i-2),Fi(10,3*i-1),0.02,'b')
set(hi,'MaxHeadSize',0,'linewidth',1) 
end
