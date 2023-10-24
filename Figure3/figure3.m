% for paper "Inferring stochastic dynamics of complex systems", T.-T. Gao,
% B. Barzel, and G. Yan
clc, clf, clear, close all

%% original data
data2norm = csvread('flocks_timeseries_2dim_hf4.csv');
data2v = (data2norm(2:end,:)-data2norm(1:end-1,:))./0.01;

figure;
for i = 1:size(data2norm,2)/2
    plot(data2norm(1:20000,i*2-1),data2norm(1:20000,i*2),'linewidth',2)
    hold on
end
N = size(data2norm,2)/2;
dt = 0.01;
Time = length(data2norm);
record_time = 100;
end_time = 1;
Nnodes = N;
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

s = normrnd(0,sqrt(dt),Time,Nnodes*dim)*5e-5;

A = ones(Nnodes,Nnodes);
for i = 1:Nnodes
    A(i,i) = 0;
end
R = zeros(Time,dim*Nnodes);
V = zeros(Time,dim*Nnodes);
dVdt = zeros(Time,dim*Nnodes);
R(1,:) = data2norm(1,:);
V(1,:) = (data2norm(2,:)-data2norm(1,:))/dt;

% scaling coefficients
%%%%%%%%%%%%%%%%%hf1%%%%%%%%%%%%%%%%%%%%%%%%
%cohesion
% p = -0.0070417263;
% b = 0.9400337338447571;
% c = 0.000451006;
% %align
% d = 0.60338336;
% e = -0.9954246;
% f = -0.0032163374;
% %self
% g = 1.9591489127252002;
% m = -1.2232491144459345e-06;
% r= -1.5962869e-05;
%%%%%%%%%%%%%%%%%hf2%%%%%%%%%%%%%%%%%%%%%%%%%%
% %cohesion
% p = 0.18648678;
% b = 0.9999883770942688;
% c = 0.0037961379;
% %align
% d = -0.20306803;
% e = 0.99999917;
% f = -4.2576343e-05;
% %self
% g = 6.167326581717668;
% m = -1.331904513790505e-06;
% r= -5.1870942e-05;
% %%%%%%%%%%%%%%%%%hf3%%%%%%%%%%%%%%%%%%%%%%%%%
%cohesion
% p = -0.106570505;
% b = 0.9631454944610596;
% c = 0.00034091622;
% %align
% d = 0.004008455;
% e = -0.99722046;
% f = 0.0011027143;
% %self
% g = 0.29124801565443115;
% m = -7.389401162072318e-07;
% r= -5.2824616e-06;
%%%%%%%%%%%%%%%%%%%%hf4%%%%%%%%%%%%%%%%%%%%%
p = -0.01368125;
b = 0.9843001;
c = 0.0025799908;
d = 0.1316837;
e = -0.9988282;
f = 0.00062561035;
g = 1.7975674649230606;
m = -8.477218784719298e-07;
r= 7.459521e-05;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for t = 2:Time
    for i = 1:Nnodes
        tmp = zeros(dim,1);
        for j = 1:Nnodes
            rij = sqrt((R(t-1,2*i-1)-R(t-1,2*j-1))^2+(R(t-1,2*i)-R(t-1,2*j))^2);
            Rij = [R(t-1,2*j-1)-R(t-1,2*i-1);R(t-1,2*j)-R(t-1,2*i);];
            Vij = [V(t-1,2*j-1)-V(t-1,2*i-1);V(t-1,2*j)-V(t-1,2*i);];
            if rij <= 0.1
                tmp = tmp+A(i,j)*((p*((rij/2-1)^3/(rij/2+1)^6+b)+c).*Rij+...
                    (d*(exp(-rij/3)+e)+f).*Vij);
            else
                tmp = tmp+A(i,j)*((p*((rij/2-1)^3/(rij/2+1)^6+b)+c).*Rij.*0+...
                    (d*(exp(-rij/3)+e)+f).*Vij.*0);
            end
        end
        Vi = [V(t-1,2*i-1);V(t-1,2*i);];
        S = [s(t,2*i-1);s(t,2*i)];
        vi = sqrt(V(t-1,2*i-1)^2+V(t-1,2*i)^2);
        a = (g*(vi+m)+r).*Vi+tmp;
        dVdt(t-1,2*i-1:2*i) = a';
        V(t,2*i-1:2*i) = V(t-1,2*i-1:2*i)+a'*dt+S';
        R(t,2*i-1:2*i) = R(t-1,2*i-1:2*i)+V(t-1,2*i-1:2*i)*dt;
    end
end

figure;
for i = 1:Nnodes
    plot(R(1:200,2*i-1),R(1:200,2*i),'--','LineWidth',2,'Color',[0.80392 0.36078 0.36078])
    hold on
    plot(data2norm(1:200,2*i-1),data2norm(1:200,2*i),'--','LineWidth',2,'Color',[0.82745 0.82745 0.82745])
end

Rv = (R(2:end,:)-R(1:end-1,:))./0.01;

TT = 25;
figure;
for i = 1:Nnodes
h = quiver(R(TT,2*i-1),R(TT,2*i),Rv(TT,2*i-1),Rv(TT,2*i),0.05,'k')
set(h,'MaxHeadSize',50) 
hold on
hi = quiver(data2norm(TT,2*i-1),data2norm(TT,2*i),data2v(TT,2*i-1),data2v(TT,2*i),0.05,'b')
set(hi,'MaxHeadSize',50) 
end

%%
RRi = (-1:0.01:1);
RRj = (-1:0.01:1);
for i = 1:length(RRi)
    for j = 1:length(RRj)
    rij = sqrt((RRi(i))^2+(RRj(j))^2);
    Rij = [RRi(i);RRj(j);];
    tmp = p*((rij/2-1)^3/(rij/2+1)^6+b)+c;
    tmp1 = d*(exp(-rij/3)+e)+f;
    end
end


%colormapFlag = [242,240,247;218,218,235;188,189,220;158,154,200;117,107,177;84,39,143;]./255;
colormapFlag = [254,235,226;252,197,192;250,159,181;247,104,161;197,27,138;122,1,119;]./255;


X = linspace(1,640,6);
colormapFlag = interp1(X,colormapFlag,linspace(1,640,640));

%% align and cohesion function plot
figure;
[X,Y] = meshgrid(linspace(-0.1,0.1,100),linspace(-0.1,0.1,100));
Z1 = (p.*(((sqrt(X.^2+Y.^2))./2-1).^3./((sqrt(X.^2+Y.^2))./2+1).^6+b)+c).*X;
Z2 = d.*(exp(-sqrt(X.^2+Y.^2)./3)+e)+f;
surf(X,Y,Z2)
%surf(X,Y,Z1,'linestyle','none');
% set(gca,'xtick',[],'xticklabel',[])
% set(gca,'ytick',[],'yticklabel',[])
% set(gca,'ztick',[],'zticklabel',[])
colormap(colormapFlag)

figure;
% real viscek
[X,Y] = meshgrid(linspace(-1,1,100),linspace(-1,1,100));
Z1 = 1.5.*((1-(sqrt(X.^2+Y.^2)/2).^3)./(1+(sqrt(X.^2+Y.^2)./2).^6)).*X;
Z2 = 1.*exp(-sqrt(X.^2+Y.^2)./3);
surf(X,Y,Z2);
colormap(colormapFlag)

%% Original Vicsek comparison
% 
% 
% R = zeros(Time,dim*Nnodes);
% V = zeros(Time,dim*Nnodes);
% dVdt = zeros(Time,dim*Nnodes);
% R(1,:) = data2norm(1,:);
% V(1,:) = (data2norm(2,:)-data2norm(1,:))/dt;
% 
% gamma = 2;
% v0 = 1.5;
% epsilon0 = 1.5;
% r0 = 2;
% epsilon1 = 1;
% r1 = 3;
% sigma = 1;
% 
% dt = 0.01;
% 
% 
% for t = 2:Time
%     for i = 1:Nnodes
%         tmp = zeros(2,1);
%         for j = 1:Nnodes
%             rij = sqrt((R(t-1,2*i-1)-R(t-1,2*j-1))^2+(R(t-1,2*i)-R(t-1,2*j))^2);
%             Rij = [R(t-1,2*j-1)-R(t-1,2*i-1);R(t-1,2*j)-R(t-1,2*i);];
%             Vij = [V(t-1,2*j-1)-V(t-1,2*i-1);V(t-1,2*j)-V(t-1,2*i);];
%             tmp = tmp+A(i,j)*((epsilon0*((1-(rij/r0)^3)/(1+(rij/r0)^6))).*Rij+...
%                 epsilon1*exp(-rij/r1).*Vij);
%             
%         end
%         Vi = [V(t-1,2*i-1);V(t-1,2*i);];
%         S = [s(t,2*i-1);s(t,2*i)];
%         a = gamma*(v0^2-(V(t-1,2*i-1)^2+V(t-1,2*i)^2)).*Vi+tmp;
%         dVdt(t-1,2*i-1:2*i) = a';
%         V(t,2*i-1:2*i) = V(t-1,2*i-1:2*i)+a'*dt+(S.*Vi)';
%         R(t,2*i-1:2*i) = R(t-1,2*i-1:2*i)+V(t-1,2*i-1:2*i)*dt;
%     end
% end
% 
% figure;
% for i = 1:Nnodes
%     plot(R(:,2*i-1),R(:,2*i),'LineWidth',1)
%     hold on
% end
% 
% figure;
% for i = 1:Nnodes
% h = quiver(R(500,2*i-1),R(500,2*i),Rv(500,2*i-1),Rv(500,2*i),0.05)
% set(h,'MaxHeadSize',50) 
% hold on
% end
% 
% %%
% figure;
% for i = 1:Nnodes
%     plot(R(1:200,2*i-1),R(1:200,2*i),'--','LineWidth',2,'Color',[0.80392 0.36078 0.36078])
%     hold on
%     plot(data2norm(1:200,2*i-1),data2norm(1:200,2*i),'--','LineWidth',2,'Color',[0.82745 0.82745 0.82745])
% end