% for paper "Inferring stochastic dynamics of complex systems", T.-T. Gao,
% B. Barzel, and G. Yan

% Equation from paper "Inferring the dynamics of underdamped stochastic
% systems" by D. B. Bruckner, et al.

clc, clf, clear, close all
tic

Nnodes = 20;
dim = 3;

A = ones(Nnodes,Nnodes);
for i = 1:Nnodes
    A(i,i) = 0;
end

gamma = 2;
v0 = 1.5;
epsilon0 = 1.5;
r0 = 2;
epsilon1 = 1;
r1 = 3;
sigma = 1;

dt = 0.01;

rs = csvread('simulation_flocks_timeseries_20new.csv');
R0 = rs(1,:);
%load('Rmatrix.mat');
%R0 = reshape(Rmatrix',[1 Nnodes*3]);
V0 = zeros(1,dim*Nnodes);


Time = 1000;
record_time = 1;
end_time = 1000;

s = normrnd(0,sqrt(dt),Time,Nnodes*dim)*5;

R = zeros(Time,dim*Nnodes);
V = zeros(Time,dim*Nnodes);
dVdt = zeros(Time,dim*Nnodes);
R(1,:) = R0;
V(1,:) = V0;

%% 2-dimensional Vicsek flocks
% for t = 2:Time
%     for i = 1:Nnodes
%         tmp = zeros(dim,1);
%         for j = 1:Nnodes
%             %rij = sqrt((R(t-1,3*i-2)-R(t-1,3*j-2))^2+(R(t-1,3*i-1)-R(t-1,3*j-1))^2+...
%             %(R(t-1,3*i)-R(t-1,3*j))^2);
%             rij = sqrt((R(t-1,2*i-1)-R(t-1,2*j-1))^2+(R(t-1,2*i)-R(t-1,2*j))^2);
%             
%             %Rij = [R(t-1,3*j-2)-R(t-1,3*i-2);R(t-1,3*j-1)-R(t-1,3*i-1);R(t-1,3*j)-R(t-1,3*i);];
%             Rij = [R(t-1,2*j-1)-R(t-1,2*i-1);R(t-1,2*j)-R(t-1,2*i);];
%             %Vij = [V(t-1,3*j-2)-V(t-1,3*i-2);V(t-1,3*j-1)-V(t-1,3*i-1);V(t-1,3*j)-V(t-1,3*i);];
%             Vij = [V(t-1,2*j-1)-V(t-1,2*i-1);V(t-1,2*j)-V(t-1,2*i);];
%             tmp = tmp+A(i,j)*((epsilon0*((1-(rij/r0)^3)/(1+(rij/r0)^6))).*Rij+...
%                 epsilon1*exp(-rij/r1).*Vij);
%         end
%         %Vi = [V(t-1,3*i-2);V(t-1,3*i-1);V(t-1,3*i);];
%         Vi = [V(t-1,2*i-1);V(t-1,2*i);];
%         %S = [s(t,3*i-2);s(t,3*i-1);s(t,3*i)];
%         S = [s(t,2*i-1);s(t,2*i);];
%         %a = gamma*(v0^2-(V(t-1,3*i-2)^2+V(t-1,3*i-1)^2+V(t-1,3*i)^2)).*Vi+tmp;
%         a = gamma*(v0^2-(V(t-1,2*i-1)^2+V(t-1,2*i)^2)).*Vi+tmp;
%         %dVdt(t-1,3*i-2:3*i) = a';
%         dVdt(t-1,2*i-1:2*i) = a';
%         %V(t,3*i-2:3*i) = V(t-1,3*i-2:3*i)+a'*dt+S';
%         V(t,2*i-1:2*i) = V(t-1,2*i-1:2*i)+a'*dt+S';
%         %R(t,3*i-2:3*i) = R(t-1,3*i-2:3*i)+V(t-1,3*i-2:3*i)*dt;
%         R(t,2*i-1:2*i) = R(t-1,2*i-1:2*i)+V(t-1,2*i-1:2*i)*dt;
%     end
% end
% figure;
% for i = 1:Nnodes
%     plot(R(record_time:end_time,2*i-1),R(record_time:end_time,2*i),'LineWidth',1)
%     hold on
% end
% 
% data2v = (R(2:end,:)-R(1:end-1,:))./0.01;
% dvdt = (data2v(2:end,:)-data2v(1:end-1,:))./0.01;
% 
% figure;
% for i = 1:100
% h = quiver(R(104,2*i-1),R(104,2*i),data2v(105,2*i-1),data2v(105,2*i),0.01,'linewidth',1)
% set(h,'MaxHeadSize',5)
% %axis equal
% hold on
% end
% figure;
% for i = 1:100
% h = quiver(R(104,2*i-1),R(104,2*i),dvdt(105,2*i-1),dvdt(105,2*i),0.0002,'linewidth',1)
% set(h,'MaxHeadSize',3) 
% hold on
% end

%% 3-dimensional Vicsek model
for t = 2:Time
    for i = 1:Nnodes
        tmp = zeros(3,1);
        for j = 1:Nnodes
            rij = sqrt((R(t-1,3*i-2)-R(t-1,3*j-2))^2+(R(t-1,3*i-1)-R(t-1,3*j-1))^2+...
            (R(t-1,3*i)-R(t-1,3*j))^2);
            Rij = [R(t-1,3*j-2)-R(t-1,3*i-2);R(t-1,3*j-1)-R(t-1,3*i-1);R(t-1,3*j)-R(t-1,3*i);];
            Vij = [V(t-1,3*j-2)-V(t-1,3*i-2);V(t-1,3*j-1)-V(t-1,3*i-1);V(t-1,3*j)-V(t-1,3*i);];
            tmp = tmp+A(i,j)*((epsilon0*((1-(rij/r0)^3)/(1+(rij/r0)^6))).*Rij+...
                epsilon1*exp(-rij/r1).*Vij);
            
        end
        Vi = [V(t-1,3*i-2);V(t-1,3*i-1);V(t-1,3*i);];
        S = [s(t,3*i-2);s(t,3*i-1);s(t,3*i)];
        a = gamma*(v0^2-(V(t-1,3*i-2)^2+V(t-1,3*i-1)^2+V(t-1,3*i)^2)).*Vi+tmp;
        dVdt(t-1,3*i-2:3*i) = a';
        V(t,3*i-2:3*i) = V(t-1,3*i-2:3*i)+a'*dt+(S.*Vi)';
        R(t,3*i-2:3*i) = R(t-1,3*i-2:3*i)+V(t-1,3*i-2:3*i)*dt;
    end
end

figure;
for i = 1:Nnodes
    plot3(R(record_time:end_time,3*i-2),R(record_time:end_time,3*i-1),R(record_time:end_time,3*i),'LineWidth',1)
    hold on
end

for i = 1:Nnodes
    X1(:,i) = R(:,3*i-2);
    X2(:,i) = R(:,3*i-1);
    X3(:,i) = R(:,3*i);
end
% figure;
% plotstarlings(X1,X2,X3,Time)


% writematrix(A, 'flocks_A_20new.csv');
% writematrix(R(record_time:end,:),'flocks_timeseries_20new.csv');


%colormapFlag = [242,240,247;218,218,235;188,189,220;158,154,200;117,107,177;84,39,143;]./255;
colormapFlag = [254,235,226;252,197,192;250,159,181;247,104,161;197,27,138;122,1,119;]./255;
% colormapFlag = [236,231,242;...
%     208,209,230;...
%     166,189,219;...
%     54,144,192;...
%     5,112,176;...
%     4,90,141;]./255;
X = linspace(1,640,6);
colormapFlag = interp1(X,colormapFlag,linspace(1,640,640));

figure(3)
record_time = 100;
colormap(colormapFlag)
N = length(R(record_time:end,1));
t = (1:1:N);
for i = 1:length(A)
patch(R(record_time:end,3*i-2),R(record_time:end,3*i-1),R(record_time:end,3*i),t,'FaceColor','None','EdgeColor','interp','linewidth',2)
%colorbar
view(3)
end


figure(2)
colormap(colormapFlag)
N = length(rs(record_time:end,1));
t = (1:1:N);
for i = 1:length(A)
patch(rs(record_time:end,3*i-2),rs(record_time:end,3*i-1),rs(record_time:end,3*i),t,'FaceColor','None','EdgeColor','interp','linewidth',1)
%colorbar
view(3)
end
%% inferred
for t = 2:Time
    for i = 1:Nnodes
        tmp = zeros(3,1);
        for j = 1:Nnodes
            rij = sqrt((R(t-1,3*i-2)-R(t-1,3*j-2))^2+(R(t-1,3*i-1)-R(t-1,3*j-1))^2+...
            (R(t-1,3*i)-R(t-1,3*j))^2);
            Rij = [R(t-1,3*j-2)-R(t-1,3*i-2);R(t-1,3*j-1)-R(t-1,3*i-1);R(t-1,3*j)-R(t-1,3*i);];
            Vij = [V(t-1,3*j-2)-V(t-1,3*i-2);V(t-1,3*j-1)-V(t-1,3*i-1);V(t-1,3*j)-V(t-1,3*i);];
            tmp = tmp+A(i,j)*(([1.3344506;1.3472373;1.3385607]*((1-(rij/r0)^3)/(1+(rij/r0)^6))).*Rij+...
                [0.9980929;1.0017698;0.9999031;]*exp(-rij/r1).*Vij);
            
        end
        Vi = [V(t-1,3*i-2);V(t-1,3*i-1);V(t-1,3*i);];
        S = [s(t,3*i-2);s(t,3*i-1);s(t,3*i)];
        a = 4.026547-1.9714836*(V(t-1,3*i-2)^2+V(t-1,3*i-1)^2+V(t-1,3*i)^2).*Vi+tmp;
        dVdt(t-1,3*i-2:3*i) = a';
        V(t,3*i-2:3*i) = V(t-1,3*i-2:3*i)+a'*dt+S';
        R(t,3*i-2:3*i) = R(t-1,3*i-2:3*i)+V(t-1,3*i-2:3*i)*dt;
    end
end

figure;
for i = 1:Nnodes
    plot3(R(record_time:end_time,3*i-2),R(record_time:end_time,3*i-1),R(record_time:end_time,3*i),'LineWidth',1)
    hold on
end

for i = 1:Nnodes
    X1(:,i) = R(:,3*i-2);
    X2(:,i) = R(:,3*i-1);
    X3(:,i) = R(:,3*i);
end
% figure;
% plotstarlings(X1,X2,X3,Time)


%writematrix(A, 'flocks_A_5.csv');
%writematrix(R(record_time:end,:),'flocks_timeseries_5.csv');


%colormapFlag = [242,240,247;218,218,235;188,189,220;158,154,200;117,107,177;84,39,143;]./255;
colormapFlag = [254,235,226;252,197,192;250,159,181;247,104,161;197,27,138;122,1,119;]./255;
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
N = length(R(record_time:end,1));
t = (1:1:N);
for i = 1:length(A)
patch(R(record_time:end,3*i-2),R(record_time:end,3*i-1),R(record_time:end,3*i),t,'FaceColor','None','EdgeColor','interp','linewidth',1)
%colorbar
view(3)
end
