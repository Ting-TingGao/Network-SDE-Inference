% for paper "Inferring stochastic dynamics of complex systems", T.-T. Gao,
% B. Barzel, and G. Yan
clc; clear;
%load data
data = csvread('PathDataNewValues.csv');
injection = csvread('injection_position.csv');

% NTG
Type1 = zeros(4,size(data,2));
Type1(1,:) = mean(data(1:4,1:end));
std1 = std(data(1:4,1:end));
Type1(2,:) = mean(data(8:15,1:end));
std2 = std(data(8:15,1:end));
Type1(3,:) = mean(data(23:28,1:end));
std3 = std(data(23:28,1:end));
Type1(4,:) = mean(data(34:39,1:end));
std4 = std(data(34:39,1:end));

% G2019S
% Type2 = zeros(4,size(data,2));
% Type2(1,:) = mean(data(5:7,1:end));
% std1 = std(data(5:7,1:end));
% Type2(2,:) = mean(data(16:22,1:end));
% std2 = std(data(16:22,1:end));
% Type2(3,:) = mean(data(29:33,1:end));
% std3 = std(data(29:33,1:end));
% Type2(4,:) = mean(data(40:end,1:end));
% std4 = std(data(40:end,1:end));

% data augumentation
DataSet = zeros(5000,size(data,2)*5);
for i = 1:5000
    m1 = round(rand(1,2).*3+1);
    m2 = round(rand(1,4).*7+8);
    m3 = round(rand(1,3).*5+23);
    m4 = round(rand(1,3).*5+34);
%     DataSet(4*i-3,1:size(data,2)) = injection;
%     DataSet(4*i-3,size(data,2)+1) = 1;
%     DataSet(4*i-3,size(data,2)+2:end) = std1.*randn(1,160).*0.1 + Type1(1,:);
%     DataSet(4*i-2,1:size(data,2)) = injection;
%     DataSet(4*i-2,size(data,2)+1) = 3;
%     DataSet(4*i-2,size(data,2)+2:end) = std2.*randn(1,160).*0.1 + Type1(2,:);
%     DataSet(4*i-1,1:size(data,2)) = injection;
%     DataSet(4*i-1,size(data,2)+1) = 3;
%     DataSet(4*i-1,size(data,2)+2:end) = std3.*randn(1,160).*0.1 + Type1(3,:);
%     DataSet(4*i,1:size(data,2)) = injection;
%     DataSet(4*i,size(data,2)+1) = 3;
%     DataSet(4*i,size(data,2)+2:end) = std4.*randn(1,160).*0.1 + Type1(4,:);
    DataSet(i,1:size(data,2)) = injection;
    DataSet(i,size(data,2)+1:size(data,2)*2) = std1.*randn(1,160).*0.1 + Type1(1,:);
    DataSet(i,size(data,2)*2+1:size(data,2)*3) = std2.*randn(1,160).*0.08 + Type1(2,:);
    DataSet(i,size(data,2)*3+1:size(data,2)*4) = std3.*randn(1,160).*0.08 + Type1(3,:);
    DataSet(i,size(data,2)*4+1:size(data,2)*5) = std4.*randn(1,160).*0.08 + Type1(4,:);
end

%writematrix(DataSet,'initial_injection_dataset.csv');

% load('W.mat');
% %ic*ic -> adjust the matrix
% Wnew = zeros(length(W));
% Wnew(1:length(W)/2,1:length(W)/2) = W(1:length(W)/2,1:length(W)/2);
% Wnew(length(W)/2+1:end,length(W)/2+1:end) = W(length(W)/2+1:end,1:length(W)/2);
% Wnew(length(W)/2+1:end,1:length(W)/2) = W(length(W)/2+1:end,length(W)/2+1:end);
% Wnew(1:length(W)/2,length(W)/2+1:end) = W(1:length(W)/2,length(W)/2+1:end);

A = csvread('TauWValues.csv');
ii = 1;
for i = 1:size(A,1)
    for j = 1:size(A,1)
        if A(i,j) ~= 0
            E(ii,1) = i;
            E(ii,2) = j;
            E(ii,3) = A(i,j);
            ii = ii+1;
        end
    end
end
Enew = sortrows(E,2);

%DataSet = csvread('path_data_augment.csv');
STD = [std1;std2;std3;std4];


% retro A; antero A';
aA = A';
ii = 1;
for i = 1:size(aA,1)
    for j = 1:size(aA,1)
        if aA(i,j) ~= 0
            aE(ii,1) = i;
            aE(ii,2) = j;
            aE(ii,3) = aA(i,j);
            ii = ii+1;
        end
    end
end
aEnew = sortrows(aE,2);

RA_E = [];
RA_E(1:length(aEnew),1:3) = Enew;

for i = 1:length(aEnew)
    m = find(Enew(:,1)==aEnew(i,1));
    n = find(Enew(m,2)==aEnew(i,2));
    if length(m) ~= 0 && length(n) ~= 0
        RA_E(m(n),4) = aEnew(i,3);
    else
        tmp = [aEnew(i,1:2),0,aEnew(i,3)];
        RA_E = [RA_E;tmp];
    end
end
Ret_Ant_E = sortrows(RA_E,2);
kk = Ret_Ant_E(:,3:4);
tempE = [Ret_Ant_E(:,1:2),kk];
%writematrix(tempE,'retro_antero_edge.csv');

% writematrix(DataSet,'augDataSet_ave.csv');

% writematrix(Type1,'aveData.csv');

 D = csvread('TauD.csv');
 Dnew = 1./log(D.^2);
 ii = 1;
for i = 1:size(Dnew,1)
    for j = 1:size(Dnew,1)
        if Dnew(i,j) ~= 0 && Dnew(i,j)>=0.11
            DE(ii,1) = i;
            DE(ii,2) = j;
            DE(ii,3) = Dnew(i,j);
            ii = ii+1;
        end
    end
end
DEnew = sortrows(DE,2);

RA_ED = [];
RA_ED(1:length(Ret_Ant_E),1:4) = Ret_Ant_E;

for i = 1:length(DEnew)
    m = find(Ret_Ant_E(:,1)==DEnew(i,1));
    n = find(Ret_Ant_E(m,2)==DEnew(i,2));
    if length(m) ~= 0 && length(n) == 1
        RA_ED(m(n),5) = DEnew(i,3);
    else
        tmp = [DEnew(i,1:2),0,0,DEnew(i,3)];
        RA_ED = [RA_ED;tmp];
    end
end
Ret_Ant_ED = sortrows(RA_ED,2);

%writematrix(Ret_Ant_ED,'Elist_re_an_eu.csv');
Dnew(Dnew<0.11)=0;

%%
t = [1,3,6,9];
Functions = zeros(4,160);
for i = 1:160
    tmp_r = 0;
    tmp_a = 0;
    tmp_e = 0;
    for j = 1:160
%         tmp_r = tmp_r + (injection(j)./(1+exp(-A(j,i))));
%         tmp_a = tmp_a + (injection(j)./(1+exp(-aA(j,i))));
%         tmp_e = tmp_e + (injection(j)*exp(Dnew(j,i)));
        if A(j,i)~=0
            tmp_r = tmp_r + (injection(j)./(1+exp(-A(j,i))));
        else
            tmp_r = tmp_r + 0;
        end
        if aA(j,i)~=0
            tmp_a = tmp_a + (injection(j)./(1+exp(-aA(j,i))));
        else
            tmp_a = tmp_a + 0;
        end
        if Dnew(j,i) ~=0
            tmp_e = tmp_e + (injection(j)*exp(Dnew(j,i)));
        else
            tmp_e = tmp_e + 0;
        end
    end
    Function(1,i) = injection(i);
    Function(2,i) = tmp_r;
    Function(3,i) = tmp_a;
    Function(4,i) = tmp_e;
end

tt = [-1.5632; -6.1588; -18.3199; -27.8956;]+[1.5*1;1.5*3;1.5*6;1.5*9;]; % c = ct+1.5t
 

%% tau diffusion prediction
coef = csvread('inferred_coef_NTG.csv');
%coef = csvread('inferred_coef_mutation_re08_ant028.csv'); % for mutation system
y_infer = [];

for i = 1:160
    tmp_r = 0;
    tmp_a = 0;
    tmp_e = 0;
    for j = 1:160
        if A(j,i)~=0
            tmp_r = tmp_r + (injection(j)./(1+exp(-A(j,i))));
        else
            tmp_r = tmp_r + 0;
        end
        if aA(j,i)~=0
            tmp_a = tmp_a + (injection(j)./(1+exp(-aA(j,i))));
        else
            tmp_a = tmp_a + 0;
        end
        if Dnew(j,i) ~=0
            tmp_e = tmp_e + (injection(j)*exp(Dnew(j,i)));
        else
            tmp_e = tmp_e + 0;
        end
    end
    y_infer(1,i) = (injection(i)*coef(1,i)+tmp_r*coef(2,i)+tmp_a*coef(3,i)+tmp_e*coef(4,i))*tt(1);
    y_infer(2,i) = (injection(i)*coef(1,i)+tmp_r*coef(2,i)+tmp_a*coef(3,i)+tmp_e*coef(4,i))*tt(2);
    y_infer(3,i) = (injection(i)*coef(1,i)+tmp_r*coef(2,i)+tmp_a*coef(3,i)+tmp_e*coef(4,i))*tt(3);
    y_infer(4,i) = (injection(i)*coef(1,i)+tmp_r*coef(2,i)+tmp_a*coef(3,i)+tmp_e*coef(4,i))*tt(4);
end
% stochastic
y_infer(1,:) = y_infer(1,:)+std1.*randn(1,160).*0.1;
y_infer(2,:) = y_infer(2,:)+std2.*randn(1,160).*0.1;
y_infer(3,:) = y_infer(3,:)+std3.*randn(1,160).*0.1;
y_infer(4,:) = y_infer(4,:)+std4.*randn(1,160).*0.1;

% determ
%y_infer(1,:) = abs(y_infer(1,:));
%y_infer(2,:) = abs(y_infer(2,:));
%y_infer(3,:) = abs(y_infer(3,:));
%y_infer(4,:) = abs(y_infer(4,:));



%% NTG plot
figure;
subplot(1,4,1)
%set(gcf, 'Position', [500,500,400,400])
plot(log(Type1(1,:)),log(y_infer(1,:)),'o','MarkerFaceColor','b')
hold on
plot([-10,2],[-10,2],'-')
xlim([-10,2]);
ylim([-10,2]); 
mdl = fitlm(Type1(1,:),y_infer(1,:))


subplot(1,4,2)
%set(gcf, 'Position', [500,500,400,400])
plot(log(Type1(2,:)),log(y_infer(2,:)),'o','MarkerFaceColor','b')
hold on
plot([-10,2],[-10,2],'-')
xlim([-10,2]);
ylim([-10,2]); 
mdl = fitlm(Type1(2,:),y_infer(2,:))

subplot(1,4,3)
%set(gcf, 'Position', [500,500,400,400])
plot(log(Type1(3,:)),log(y_infer(3,:)),'o','MarkerFaceColor','b')
hold on
plot([-10,2],[-10,2],'-')
xlim([-10,2]);
ylim([-10,2]); 
mdl = fitlm(Type1(3,:),y_infer(3,:))

subplot(1,4,4)
%set(gcf, 'Position', [500,500,400,400])
plot(log(Type1(4,:)),log(y_infer(4,:)),'o','MarkerFaceColor','b')
hold on
plot([-10,2],[-10,2],'-')
xlim([-10,2]);
ylim([-10,2]); 
mdl = fitlm(Type1(4,:),y_infer(4,:))


%% Mutation plot
% figure;
% subplot(1,4,1)
% %set(gcf, 'Position', [500,500,400,400])
% plot(log(Type2(1,:)),log(y_infer(1,:)),'o','MarkerFaceColor','b')
% hold on
% plot([-10,2],[-10,2],'-')
% xlim([-10,2]);
% ylim([-10,2]); 
% mdl = fitlm(Type2(1,:),y_infer(1,:))
% 
% 
% subplot(1,4,2)
% %set(gcf, 'Position', [500,500,400,400])
% plot(log(Type2(2,:)),log(y_infer(2,:)),'o','MarkerFaceColor','b')
% hold on
% plot([-10,2],[-10,2],'-')
% xlim([-10,2]);
% ylim([-10,2]); 
% mdl = fitlm(Type2(2,:),y_infer(2,:))
% 
% subplot(1,4,3)
% %set(gcf, 'Position', [500,500,400,400])
% plot(log(Type2(3,:)),log(y_infer(3,:)),'o','MarkerFaceColor','b')
% hold on
% plot([-10,2],[-10,2],'-')
% xlim([-10,2]);
% ylim([-10,2]); 
% mdl = fitlm(Type2(3,:),y_infer(3,:))
% 
% subplot(1,4,4)
% %set(gcf, 'Position', [500,500,400,400])
% plot(log(Type2(4,:)),log(y_infer(4,:)),'o','MarkerFaceColor','b')
% hold on
% plot([-10,2],[-10,2],'-')
% xlim([-10,2]);
% ylim([-10,2]); 
% mdl = fitlm(Type2(4,:),y_infer(4,:))
%

%% null model test
% rsquare = zeros(500,4);
% Error = zeros(4,500);
% for nt = 1:500
% y_infer = [];
% injection = zeros(1,160);
% %idx = randi(160,5,1);
% idx = [52,60,62,63,96];
% %idx = [60,62,63,96];
% injection(idx)=1;
% for i = 1:160
%     tmp_r = 0;
%     tmp_a = 0;
%     tmp_e = 0;
%     for j = 1:160
%         tmp_r = tmp_r + (injection(j)./(1+exp(-A(j,i))));
%         tmp_a = tmp_a + (injection(j)./(1+exp(-aA(j,i))));
%         tmp_e = tmp_e + (injection(j)*exp(Dnew(j,i)));
%     end
%     y_infer(1,i) = (injection(i)*coef(1,i)+tmp_r*coef(2,i)+tmp_a*coef(3,i)+tmp_e*coef(4,i))*tt(1);
%     y_infer(2,i) = (injection(i)*coef(1,i)+tmp_r*coef(2,i)+tmp_a*coef(3,i)+tmp_e*coef(4,i))*tt(2);
%     y_infer(3,i) = (injection(i)*coef(1,i)+tmp_r*coef(2,i)+tmp_a*coef(3,i)+tmp_e*coef(4,i))*tt(3);
%     y_infer(4,i) = (injection(i)*coef(1,i)+tmp_r*coef(2,i)+tmp_a*coef(3,i)+tmp_e*coef(4,i))*tt(4);
%     
% end
% y_infer(1,:) = y_infer(1,:)+std1.*randn(1,160).*0.1;
% y_infer(2,:) = y_infer(2,:)+std2.*randn(1,160).*0.1;
% y_infer(3,:) = y_infer(3,:)+std3.*randn(1,160).*0.1;
% y_infer(4,:) = y_infer(4,:)+std4.*randn(1,160).*0.1;
% 
% y_infer(1,:) = abs(y_infer(1,:));
% y_infer(2,:) = abs(y_infer(2,:));
% y_infer(3,:) = abs(y_infer(3,:));
% y_infer(4,:) = abs(y_infer(4,:));
% 
% error_tmp = sum(abs(Type1-y_infer),2);
% 
% mdl1 = fitlm(Type1(1,:),y_infer(1,:));
% mdl2 = fitlm(Type1(2,:),y_infer(2,:));
% mdl3 = fitlm(Type1(3,:),y_infer(3,:));
% mdl4 = fitlm(Type1(4,:),y_infer(4,:));
% 
% rsquare(nt,1) = mdl1.Rsquared.Ordinary;
% rsquare(nt,2) = mdl2.Rsquared.Ordinary;
% rsquare(nt,3) = mdl3.Rsquared.Ordinary;
% rsquare(nt,4) = mdl4.Rsquared.Ordinary;
% 
% Error(:,nt) = error_tmp;
% 
% 
% end

%% SI plot
name_id = fopen('names.csv');
Names = textscan(name_id,'%s%d%f%d','Delimiter',',');
fclose(name_id);

tmpy = csvread('inferred_tau_path.csv');
aa = load('namelist.mat');
aa = aa.aa;
tmpy = y_infer;
figure;
set(gcf, 'Position', [500,500,600,400])
xd = [1,2,3,4];
for i = 121:160 %change the range to get the different rigions' results
tmpstd = [std1(i);std2(i);std3(i);std4(i);];
%Infer = [Type1(:,i),tmpstd,tmpy(:,i)];
row_index = ceil(i/8);
column_index = i-(row_index-1)*8;
subplot(5,8,i-120)
e = errorbar(xd,Type1(:,i),tmpstd,'.','linewidth',1)
e.Color = 'blue';
hold on
plot(xd,Type1(:,i),'b-','linewidth',1)
plot(tmpy(:,i),'-','linewidth',1)
set(gca,'xticklabel',[])
title(aa{i})
end
