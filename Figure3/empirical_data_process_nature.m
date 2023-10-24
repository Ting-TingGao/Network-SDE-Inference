%% Pre-process the flocking data
% for paper "Inferring stochastic dynamics of complex systems", T.-T. Gao,
% B. Barzel, and G. Yan

clc, clf, clear, close all
tic

% timestamp, x, y, z, dxdt, dydt, dzdt, dxdt2, dydt2, dzdt2, gps_con
N = 9; % birds number
% load the data from different datasets: hf1, hf2,hf3, and hf4
data_A = csvread('hf1_A_new.csv');
data_C = csvread('hf1_C_new.csv');
data_D = csvread('hf1_D_new.csv');
data_F = csvread('hf1_F_new.csv');
data_G = csvread('hf1_G_new.csv');
data_H = csvread('hf1_H_new.csv');
data_I = csvread('hf1_I_new.csv');
data_K = csvread('hf1_K_new.csv');
data_L = csvread('hf1_L_new.csv');

% align the data
start_time = [data_A(1,1),data_C(1,1),data_D(1,1),data_F(1,1),data_G(1,1),...
    data_H(1,1),data_I(1,1),data_K(1,1),data_L(1,1)];
start_time = max(start_time);

end_time = [data_A(end,1),data_C(end,1),data_D(end,1),data_F(end,1),data_G(end,1),...
    data_H(end,1),data_I(end,1),data_K(end,1),data_L(end,1)];
end_time = min(end_time);

datacell = {data_A,data_C,data_D,data_F,data_G,data_H,data_I,data_K,data_L};


DATA = [];
for i = 1:N
    [m1,n1] = find(datacell{i}==start_time);
    [m2,n2] = find(datacell{i}==end_time);
    DATA(:,3*i-2:3*i) = datacell{i}(m1:m2,2:4);
end


figure;
for i = 1:N
    plot3(DATA(:,3*i-2),DATA(:,3*i-1),DATA(:,3*i),'LineWidth',2)
    hold on
end

%% align
DATAnew = DATA(1000:5000,:);
for i = 1:N
    x(:,i) = DATAnew(:,3*i-2);
    y(:,i) = DATAnew(:,3*i-1);
    z(:,i) = DATAnew(:,3*i);
end
xori = min(min(x));
yori = min(min(y));
zori = min(min(z));

for i = 1:N
    DATAnew2(:,3*i-2) = DATAnew(:,3*i-2)-xori;
    DATAnew2(:,3*i-1) = DATAnew(:,3*i-1)-yori;
    DATAnew2(:,3*i) = DATAnew(:,3*i)-zori;
end


figure;
for i = 1:N
    plot3(DATAnew2(:,3*i-2),DATAnew2(:,3*i-1),DATAnew2(:,3*i),'LineWidth',2)
    hold on
end
% keep the birds stay in collective motions
DATAnew2 = [DATAnew2(:,1:12),DATAnew2(:,16:18),DATAnew2(:,22:end)];%hf1
%DATAnew2 = [DATAnew2(:,1:6),DATAnew2(:,10:12),DATAnew2(:,16:end)];%hf2
%DATAnew2 = [DATAnew2(:,1:12),DATAnew2(:,16:24)];%hf3
%DATAnew2 = [DATAnew2(:,1:12),DATAnew2(:,16:end)];%hf4

figure;
for i = 1:size(DATAnew2,2)/3
    plot3(DATAnew2(:,3*i-2),DATAnew2(:,3*i-1),DATAnew2(:,3*i),'LineWidth',2)
    hold on
end

figure;
for i = 1:size(DATAnew2,2)/3
    plot(DATAnew2(:,3*i-2),DATAnew2(:,3*i-1),'LineWidth',2)
    hold on
end
%% splined, data augmentation
rdata = DATAnew2;
tt = 0.2:0.2:size(DATAnew2,1)*0.2;
ttnew = 0.2:0.01:size(DATAnew2,1)*0.2;
rdata1 = [];
for i = 1:size(DATAnew2,2)
    rdata1(:,i) = spline(tt,rdata(:,i),ttnew);
end

figure;
plot(tt,rdata(:,1),'o',ttnew,rdata1(:,1))

figure;
for i = 1:size(DATAnew2,2)/3
    plot3(rdata1(:,3*i-2),rdata1(:,3*i-1),rdata1(:,3*i),'LineWidth',2)
    hold on
end

%% normalize
rdata2 = (rdata1-min(min(rdata1)))./(max(max(rdata1))-min(min(rdata1)));
%writematrix(rdata2,'flocks_timeseries_splined_norm.csv');
figure;
for i = 1:size(DATAnew2,2)/3
    plot3(rdata2(:,3*i-2),rdata2(:,3*i-1),rdata2(:,3*i),'LineWidth',2)
    hold on
end
%m12 = csvread('message_vicsek.csv');
%dij = sqrt(m12(:,1).^2+m12(:,2).^2+m12(:,3).^2);
%% two dimensions recorded
for i = 1:size(rdata1,2)/3
data2dim(:,2*i-1) = rdata1(:,3*i-2);
data2dim(:,2*i) = rdata1(:,3*i-1);
end
data2norm = (data2dim-min(min(data2dim)))./(max(max(data2dim))-min(min(data2dim)));
figure;
for i = 1:size(data2norm,2)/2
    plot(data2norm(:,2*i-1),data2norm(:,2*i),'LineWidth',2)
    hold on
end

%writematrix(data2norm,'flocks_timeseries_2dim_hf1.csv');

data2v = (data2norm(2:end,:)-data2norm(1:end-1,:))./0.01;
dvdt = (data2v(2:end,:)-data2v(1:end-1,:))./0.01;
figure;
for i = 1:size(data2norm,2)/2
h = quiver(data2norm(3000,2*i-1),data2norm(3000,2*i),data2v(3000,2*i-1),data2v(3000,2*i),0.1)
set(h,'MaxHeadSize',50) 
hold on
end


%% hf4 inferred dynamics
dt = 0.01;
Time = length(data2norm);
record_time = 1000;
end_time = 3000;
Nnodes = 8;
dim = 2;

% s0 = normrnd(0,sqrt(dt),Time,Nnodes*dim)*5e-5;
% s1 = normrnd(0,sqrt(dt),Time,Nnodes*dim)*1e-5;
% s2 = normrnd(0,sqrt(dt),Time,Nnodes*dim)*1e-4;
% s = [s0;s1;s2];
% for i = 1:size(s,2)
%     tmp = s(:,i);
%     s_1(:,i) = tmp(randperm(numel(tmp),Time));
% end
% s = s_1;

s = normrnd(0,sqrt(dt),Time,Nnodes*dim)*1e-5;

A = ones(Nnodes,Nnodes);
for i = 1:Nnodes
    A(i,i) = 0;
end
R = zeros(Time,dim*Nnodes);
V = zeros(Time,dim*Nnodes);
dVdt = zeros(Time,dim*Nnodes);
R(1,:) = data2norm(1,:);
V(1,:) = (data2norm(2,:)-data2norm(1,:))/dt;

p = 0.0338;
b = 1;
c = 0.00261;
d = 0.426;
e = -1;
f = 0.0022;
g = 0.0281;
m = 0.001583;
r= -0.0004;

for t = 2:Time
    for i = 1:Nnodes
        tmp = zeros(dim,1);
        for j = 1:Nnodes
            rij = sqrt((R(t-1,2*i-1)-R(t-1,2*j-1))^2+(R(t-1,2*i)-R(t-1,2*j))^2);
            Rij = [R(t-1,2*j-1)-R(t-1,2*i-1);R(t-1,2*j)-R(t-1,2*i);];
            Vij = [V(t-1,2*j-1)-V(t-1,2*i-1);V(t-1,2*j)-V(t-1,2*i);];
            tmp = tmp+A(i,j)*((p*((rij/2-1)^3/(rij/2+1)^6+b)+c).*Rij+...
                (d*(exp(-rij/3)+e)+f).*Vij);
        end
        Vi = [V(t-1,2*i-1);V(t-1,2*i);];
        S = [s(t,2*i-1);s(t,2*i)];
        vi = sqrt(V(t-1,2*i-1)^2+V(t-1,2*i)^2);
        a = (g*(-vi+m)+r).*Vi+tmp;
        dVdt(t-1,2*i-1:2*i) = a';
        if t == 1000&&8000
            V(t,2*i-1:2*i) = V(t-1,2*i-1:2*i)+rand(1,2)*1e-3;
        else
            V(t,2*i-1:2*i) = V(t-1,2*i-1:2*i)+a'*dt+S';
        end
        %V(t,2*i-1:2*i) = V(t-1,2*i-1:2*i)+a'*dt+S';
        R(t,2*i-1:2*i) = R(t-1,2*i-1:2*i)+V(t-1,2*i-1:2*i)*dt;
    end
end

figure;
for i = 1:Nnodes
    plot(R(1:20000,2*i-1),R(1:20000,2*i),'--','LineWidth',2,'Color',[0.80392 0.36078 0.36078])
    hold on
    plot(data2norm(1:20000,2*i-1),data2norm(1:20000,2*i),'--','LineWidth',2,'Color',[0.82745 0.82745 0.82745])
end

Rv = (R(2:end,:)-R(1:end-1,:))./0.01;

figure;
for i = 1:Nnodes
h = quiver(R(500,2*i-1),R(500,2*i),Rv(500,2*i-1),Rv(500,2*i),0.05)
set(h,'MaxHeadSize',50) 
hold on
end

figure;
for i = 1:8
h = quiver(data2norm(500,2*i-1),data2norm(500,2*i),data2v(500,2*i-1),data2v(500,2*i),0.05)
set(h,'MaxHeadSize',50) 
hold on
end


%% distance recording
Nnodes = size(rdata2,2)/3;
A = ones(Nnodes,Nnodes);
for i = 1:Nnodes
    A(i,i) = 0;
end

for i = 1:Nnodes
    tmp = zeros(length(rdata2),1);
    for j = 1:Nnodes
        tmp = tmp+A(i,j)*sqrt((rdata2(:,3*j-2)-rdata2(:,3*i-2)).^2+(rdata2(:,3*j-1)-rdata2(:,3*i-1)).^2+(rdata2(:,3*j)-rdata2(:,3*i)).^2);
    end
    distance(:,i) = tmp/(Nnodes-1);
end

mean_dis = mean(distance,2);
figure;
for i = 1:Nnodes
    plot3(rdata2(:,3*i-2),rdata2(:,3*i-1),distance(:,i),'LineWidth',2)
    hold on
end

% delt_v = (rdata2(2:end,:)-rdata2(1:end-1,:))./0.01;
% delt_vi = rdata2(2:end,:)-rdata2(1:end-1,:);
% for i = 1:Nnodes
%     tmp = zeros(length(delt_v),1);
%     for j = 1:Nnodes
%         tmp = tmp+A(i,j)*sqrt((delt_v(:,3*j-2)-delt_v(:,3*i-2)).^2+(delt_v(:,3*j-1)-delt_v(:,3*i-1)).^2+(delt_v(:,3*j)-delt_v(:,3*i)).^2);
%     end
%     DeltV(:,i) = tmp;%/(Nnodes-1);
% end
% figure;
% plot(DeltV(:,1),sqrt(delt_vi(:,1).^2+delt_vi(:,2).^2+delt_vi(:,3).^2),'o');
% 
% figure;
% plot(distance(2:end,1),sqrt(delt_vi(:,1).^2+delt_vi(:,2).^2+delt_vi(:,3).^2),'o');
%% mechanism of functions
% xtest = 0.01:0.01:1;
% ytest = exp(-xtest/3);
% figure;
% plot(xtest,ytest,'k','LineWidth',2);
% 
% ztest = ((xtest/2-1).^3)./((xtest/2+1).^6)+1;
% %ztest = ((xtest/2).^3-1)./((xtest/2).^6+1);
% figure;
% plot(xtest,ztest,'k','LineWidth',2);

%% order parameter
% delt_v = (rdata2(2:end,:)-rdata2(1:end-1,:))./0.01;
% 
% tmp_x = zeros(length(rdata2)-1,1);
% tmp_y = zeros(length(rdata2)-1,1);
% tmp_z = zeros(length(rdata2)-1,1);
% for i = 1:Nnodes
%     tmp_x = tmp_x+delt_v(:,3*i-2);
%     tmp_y = tmp_y+delt_v(:,3*i-1);
%     tmp_z = tmp_z+delt_v(:,3*i);
%     tmp = zeros(length(rdata2),1);
%     for j = 1:Nnodes
%          tmp = tmp+A(i,j)*sqrt((rdata2(:,3*j-2)-rdata2(:,3*i-2)).^2+(rdata2(:,3*j-1)-rdata2(:,3*i-1)).^2+(rdata2(:,3*j)-rdata2(:,3*i)).^2);
%     end
%     distance = sum(tmp,2)/(Nnodes-1);
% end
% 
% phi_1 = sqrt(tmp_x.^2+tmp_y.^2+tmp_z.^2)./(Nnodes*0.05);
% figure;
% plot(distance(2:end),phi_1,'.')

%%
delt_v = (rdata2(2:end,:)-rdata2(1:end-1,:))./0.01;
for i = 1:Nnodes
    delt_v_e(:,3*i-2:3*i) = delt_v(:,3*i-2:3*i)./sqrt(delt_v(:,3*i-2).^2+delt_v(:,3*i-1).^2+delt_v(:,3*i).^2);
end
tmp_x = zeros(length(rdata2)-1,1);
tmp_y = zeros(length(rdata2)-1,1);
tmp_z = zeros(length(rdata2)-1,1);
tmp = zeros(length(rdata2),1);
for i = 1:Nnodes
    tmp_x = tmp_x+delt_v_e(:,3*i-2);
    tmp_y = tmp_y+delt_v_e(:,3*i-1);
    tmp_z = tmp_z+delt_v_e(:,3*i);
    tmp = zeros(length(rdata2),1);
    for j = 1:Nnodes
         %tmp = tmp+A(i,j)*sqrt((rdata2(:,3*j-2)-rdata2(:,3*i-2)).^2+(rdata2(:,3*j-1)-rdata2(:,3*i-1)).^2+(rdata2(:,3*j)-rdata2(:,3*i)).^2);
         tmp = tmp+A(i,j)*sqrt((rdata2(:,3*j-2)-rdata2(:,3*i-2)).^2+(rdata2(:,3*j-1)-rdata2(:,3*i-1)).^2);
         distance(:,j) = tmp;
    end
    %distance(:,j) = tmp;
end
phi_1 = sqrt(tmp_x.^2+tmp_y.^2+tmp_z.^2)./Nnodes;
figure;
plot([distance(2:end,:),phi_1-0.9])
%writematrix(rdata2,'flocks_timeseries_3dim_hf4.csv');
