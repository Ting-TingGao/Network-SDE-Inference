%% Figure2 plot of HR and Rossler stochastic dynamics

% for paper "Inferring stochastic dynamics of complex systems", T.-T. Gao,
% B. Barzel, and G. Yan

% Code before line 159 is about HR system, and after line 159 is about Rossler system.
% If user want to generate HR(or Rossler) system's results, please comment
% out the other part.
%% HR
clc, clf, clear, close all
tic
xs = csvread('HR_stochastic_intensity01_10I.csv');%load the stochastic trajectories

E_list = textread('20nodes.txt');
E = csvread('edge_type_10I.csv');
E_list = E_list(3:length(E_list),:);
E_num = length(E_list);
E_list = E_list+1;
A = zeros(20);
for i = 1:E_num
    A(E_list(i,2),E_list(i,1))=E(i,1);
end

dim=3;
dt = 0.01;
tspan = dt:dt:500;

n1 = length(A);
n = n1*dim;

T = length(tspan);
Vsyn_ex = 2;
Vsyn_in = -1.5;
x0 = xs(1,:);
s = normrnd(0,sqrt(dt),T,n);
stochastic = 0.1;
gc = 0.15;
x = [];
x(1,:) = x0;
F = [];
F(1,:) = x0;

% generate the stochastic trajectories
for t = 1:T
    for i = 1:n1
        tmp = 0;
        for j = 1:n1
            if A(i,j) > 0
                tmp = tmp+A(i,j)*(gc*(Vsyn_ex-x(t,3*i-2)))/(1+exp(-10*(x(t,3*j-2)-1)));
            else
                tmp = tmp+abs(A(i,j))*(gc*(Vsyn_in-x(t,3*i-2)))/(1+exp(-10*(x(t,3*j-2)-1)));
                %tmp = tmp+A(i,j)*sin(x(t,3*j-2)-x(t,3*i-2));
            end
        end
        x(t+1,3*i-2) = x(t,3*i-2)+(x(t,3*i-1)-x(t,3*i-2)^3+3*x(t,3*i-2)^2-x(t,3*i)+3.24+tmp)*dt+stochastic*x(t,3*i-2)*s(t,3*i-2);
        x(t+1,3*i-1) = x(t,3*i-1)+(1-5*x(t,3*i-2)^2-x(t,3*i-1))*dt+stochastic*x(t,3*i-1)*s(t,3*i-1);
        x(t+1,3*i) = x(t,3*i)+(0.005*(4*(x(t,3*i-2)+1.6)-x(t,3*i)))*dt+stochastic*x(t,3*i)*s(t,3*i);
        %F(t+1,3*i-2) = x(t,3*i-1)-x(t,3*i-2)^3+3*x(t,3*i-2)^2-x(t,3*i)+3.24+tmp;
        %F(t+1,3*i-1) = 1-5*x(t,3*i-2)^2-x(t,3*i-1);
        %F(t+1,3*i) = 0.005*(4*(x(t,3*i-2)+1.6)-x(t,3*i));
    end
end
% generate the deterministic trajectories
x_det(1,:) = x0;
for t = 1:T
    for i = 1:n1
        tmp = 0;
        for j = 1:n1
            if A(i,j) > 0
                tmp = tmp+A(i,j)*(gc*(Vsyn_ex-x(t,3*i-2)))/(1+exp(-10*(x(t,3*j-2)-1)));
            else
                tmp = tmp+abs(A(i,j))*(gc*(Vsyn_in-x(t,3*i-2)))/(1+exp(-10*(x(t,3*j-2)-1)));
                %tmp = tmp+A(i,j)*sin(x(t,3*j-2)-x(t,3*i-2));
            end
        end
        x_det(t+1,3*i-2) = x_det(t,3*i-2)+(x_det(t,3*i-1)-x_det(t,3*i-2)^3+3*x_det(t,3*i-2)^2-x_det(t,3*i)+3.24+tmp)*dt;
        x_det(t+1,3*i-1) = x_det(t,3*i-1)+(1-5*x_det(t,3*i-2)^2-x_det(t,3*i-1))*dt;
        x_det(t+1,3*i) = x_det(t,3*i)+(0.005*(4*(x_det(t,3*i-2)+1.6)-x_det(t,3*i)))*dt;
        F(t+1,3*i-2) = x(t,3*i-1)-x(t,3*i-2)^3+3*x(t,3*i-2)^2-x(t,3*i)+3.24+tmp;
        F(t+1,3*i-1) = 1-5*x(t,3*i-2)^2-x(t,3*i-1);
        F(t+1,3*i) = 0.005*(4*(x(t,3*i-2)+1.6)-x(t,3*i));
    end
end

%% inferred HR SDE
i1 =  0.29702517;
i2 = -0.14409219;
i3 = -0.22863775;
i4 = -0.14750558;
i5 = 2.9944187;
i6 = -0.99276109;
i7 = 0.99810817;
i8 = -0.98233605;
i9 = 3.204363701;
i10 = -4.98009929;
i11 =  -0.99498259;
i12 = 0.99532704;
xi = [];
xi(1,:) = x0;
Fi = [];
Fi(1,:) = x0;
for t = 1:T
    for i = 1:n1
        tmp = 0;
        for j = 1:n1
            if A(i,j) > 0
                tmp = tmp+A(i,j)*(i1+i2*xi(t,3*i-2))/(1+exp(-10*(xi(t,3*j-2)-1)));
            else
                tmp = tmp+abs(A(i,j))*(i3+i4*xi(t,3*i-2))/(1+exp(-10*(xi(t,3*j-2)-1)));
            end
        end
        xi(t+1,3*i-2) = xi(t,3*i-2)+(i7*xi(t,3*i-1)+i6*xi(t,3*i-2)^3+i5*xi(t,3*i-2)^2+i8*xi(t,3*i)+i9+tmp)*dt+stochastic*xi(t,3*i-2)*s(t,3*i-2);
        xi(t+1,3*i-1) = xi(t,3*i-1)+(i12+i10*xi(t,3*i-2)^2+i11*xi(t,3*i-1))*dt+stochastic*xi(t,3*i-1)*s(t,3*i-1);
        xi(t+1,3*i) = xi(t,3*i)+(0.031040927+0.01878267*xi(t,3*i-2)-0.00480374*xi(t,3*i))*dt+stochastic*xi(t,3*i)*s(t,3*i);
        Fi(t+1,3*i-2) = i7*xi(t,3*i-1)+i6*xi(t,3*i-2)^3+i5*xi(t,3*i-2)^2+i8*xi(t,3*i)+i9+tmp;
        Fi(t+1,3*i-1) = i12+i10*xi(t,3*i-2)^2+i11*xi(t,3*i-1);
        Fi(t+1,3*i) =0.031040927+0.01878267*xi(t,3*i-2)-0.00480374*xi(t,3*i);
    end
end

%% force field comparison
figure(1);
for i = 1:20
h = quiver(x_det(10,3*i-2),x_det(10,3*i-1),F(10,3*i-2),F(10,3*i-1),0.02,'k')
set(h,'MaxHeadSize',30,'linewidth',0.8) 
hold on
hi = quiver(xi(10,3*i-2),xi(10,3*i-1),Fi(10,3*i-2),Fi(10,3*i-1),0.02,'b')
set(hi,'MaxHeadSize',30,'linewidth',1) 
end


%% Trajectories comparison
N = length(x(:,1));
t = (1:1:N);
 
colormapFlag = [222,235,247;198,219,239;158,202,225;107,174,214;49,130,189;8,81,156;]./255;
% colormapFlag = [236,231,242;...
%     208,209,230;...
%     166,189,219;...
%     54,144,192;...
%     5,112,176;...
%     4,90,141;]./255;
X = linspace(1,640,6);
colormapFlag = interp1(X,colormapFlag,linspace(1,640,640));

figure(2)
colormap(colormapFlag)
patch(x(:,1),x(:,2),x(:,3),t,'FaceColor','None','EdgeColor','interp','linewidth',3)
colorbar
axis off;
view(60,70)
%view(3)

N = length(xi(:,1));
t = (1:1:N);
figure(3)
colormap(colormapFlag)
patch(xi(:,1),xi(:,2),xi(:,3),t,'FaceColor','None','EdgeColor','interp','linewidth',3)
colorbar
axis off;
view(60,70)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Rossler
clc, clf, clear, close all
tic
xs = csvread('Rossler_stochastic_weighted_intensity01.csv');

dim=3;
dt = 0.01;
tspan = dt:dt:400;

A = csvread('rossler_weighted_adj.csv');

n1 = length(A);
n = n1*dim;

T = length(tspan);
eta = 0.1;
x0 = xs(1,:);
s = normrnd(0,sqrt(dt),T,n);
x = [];
x_determ = [];
x(1,:) = x0;
x_determ(1,:) = x0;
F = [];
F(1,:) = x0;
% generate stochastic trajectories
for t = 1:T
    for i = 1:n1
        tmp = 0;
        for j = 1:n1
            tmp = tmp+A(i,j)*(x(t,3*j-2)-x(t,3*i-2));
        end
        x(t+1,3*i-2) = x(t,3*i-2)+(-x(t,3*i-1)-x(t,3*i)+0.5*tmp)*dt+eta*x(t,3*i-2)*s(t,3*i-2);
        x(t+1,3*i-1) = x(t,3*i-1)+(x(t,3*i-2)+0.35*x(t,3*i-1))*dt+eta*x(t,3*i-1)*s(t,3*i-1);
        x(t+1,3*i) = x(t,3*i)+(0.2 + x(t,3*i)*(x(t,3*i-2)-5.7))*dt+eta*x(t,3*i)*s(t,3*i);
        F(t+1,3*i-2) = -x(t,3*i-1)-x(t,3*i)+0.5*tmp;
        F(t+1,3*i-1) = x(t,3*i-2)+0.35*x(t,3*i-1);
        F(t+1,3*i) = 0.2 + x(t,3*i)*(x(t,3*i-2)-5.7);
    end
end
% generate deterministic trajectories
for t = 1:T
    for i = 1:n1
        tmp = 0;
        for j = 1:n1
            tmp = tmp+A(i,j)*(x_determ(t,3*j-2)-x_determ(t,3*i-2));
        end
        x_determ(t+1,3*i-2) = x_determ(t,3*i-2)+(-x_determ(t,3*i-1)-x_determ(t,3*i)+0.5*tmp)*dt;
        x_determ(t+1,3*i-1) = x_determ(t,3*i-1)+(x_determ(t,3*i-2)+0.35*x_determ(t,3*i-1))*dt;
        x_determ(t+1,3*i) = x_determ(t,3*i)+(0.2 + x_determ(t,3*i)*(x_determ(t,3*i-2)-5.7))*dt;
    end
end

% inferred Rossler SDE
xi = [];
xi_determ = [];
xi(1,:) = x0;
xi_determ(1,:) = x0;
for t = 1:T
    for i = 1:n1
        tmp = 0;
        for j = 1:n1
            tmp = tmp+A(i,j)*(xi(t,3*j-2)-xi(t,3*i-2));
        end
        xi(t+1,3*i-2) = xi(t,3*i-2)+(-1.00289452*xi(t,3*i-1)-0.99984907*xi(t,3*i)+0.507567*tmp)*dt+eta*x(t,3*i-2)*s(t,3*i-2);
        xi(t+1,3*i-1) = xi(t,3*i-1)+(1.0004688*xi(t,3*i-2)+0.34313912*xi(t,3*i-1))*dt+eta*xi(t,3*i-1)*s(t,3*i-1);
        xi(t+1,3*i) = xi(t,3*i)+(0.210632896 + xi(t,3*i)*(0.99224356*xi(t,3*i-2)-5.65610949))*dt+0.498*xi(t,3*i)*s(t,3*i);
        xi_determ(t+1,3*i-2) = xi_determ(t,3*i-2)+(-1.00289452*xi_determ(t,3*i-1)-0.99984907*xi_determ(t,3*i)+0.507567*tmp)*dt;
        xi_determ(t+1,3*i-1) = xi_determ(t,3*i-1)+(1.0004688*xi_determ(t,3*i-2)+0.34313912*xi_determ(t,3*i-1))*dt;
        xi_determ(t+1,3*i) = xi_determ(t,3*i)+(0.210632896 + xi_determ(t,3*i)*(0.99224356*xi_determ(t,3*i-2)-5.65610949))*dt;
        Fi(t+1,3*i-2) = -1.00289452*xi_determ(t,3*i-1)-0.99984907*xi_determ(t,3*i)+0.507567*tmp;
        Fi(t+1,3*i-1) = 1.0004688*xi_determ(t,3*i-2)+0.34313912*xi_determ(t,3*i-1);
        Fi(t+1,3*i) =0.210632896 + xi_determ(t,3*i)*(0.99224356*xi_determ(t,3*i-2)-5.65610949);
    end
end
%% force field comparison
figure(1);
for i = 1:20
h = quiver(x_determ(10,3*i-2),x_determ(10,3*i-1),F(10,3*i-2),F(10,3*i-1),0.1,'k')
set(h,'MaxHeadSize',30,'linewidth',0.8) 
hold on
hi = quiver(xi(10,3*i-2),xi(10,3*i-1),Fi(10,3*i-2),Fi(10,3*i-1),0.1,'b')
set(hi,'MaxHeadSize',30,'linewidth',1) 
end
N = length(x(:,1));
t = (1:1:N);
%%% save pdf figure
set(gcf,'Units','Inches');
pos = get(gcf,'Position');
set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
filename = 'forcefieldRossler';
print(gcf,filename,'-dpdf','-r0')
close(gcf)
%% trajectories comparison
%colormapFlag = [254,235,226;252,197,192;250,159,181;247,104,161;197,27,138;122,1,119;]./255;
colormapFlag = [242,240,247;218,218,235;188,189,220;158,154,200;117,107,177;84,39,143;]./255;
% colormapFlag = [236,231,242;...
%     208,209,230;...
%     166,189,219;...
%     54,144,192;...
%     5,112,176;...
%     4,90,141;]./255;
X = linspace(1,640,6);
colormapFlag = interp1(X,colormapFlag,linspace(1,640,640));

figure(2)
colormap(colormapFlag)
patch(x(:,1),x(:,2),x(:,3),t,'FaceColor','None','EdgeColor','interp','linewidth',3)
colorbar
axis off;
view(29,33)
%view(3)

N = length(xi(:,1));
t = (1:1:N);
figure(3)
colormap(colormapFlag)
patch(xi(:,1),xi(:,2),xi(:,3),t,'FaceColor','None','EdgeColor','interp','linewidth',3)
colorbar
axis off;
view(29,33)
