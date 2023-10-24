%% Create the missing links for Rossler weighted network
% Ting-Ting Gao

clc, clf, clear, close all
tic

A = csvread('rossler_weighted_adj.csv');
E = [];
ii = 1;
for i = 1:length(A)
    for j = 1:length(A)
        if A(i,j) ~= 0
            E(ii,1) = j;
            E(ii,2) = i;
            E(ii,3) = A(i,j);
            ii = ii+1;
        end
    end
end

num_edges = sum(sum(A~=0));
p = 0.05;
id = randperm(size(E,1));
E(id(1:floor(p*num_edges)),:) = [];
Anew = zeros(length(A));
for i = 1:size(E,1)
    Anew(E(i,2),E(i,1)) = E(i,3);
end
writematrix(Anew, 'weighted_adj_miss005_2.csv');
writematrix(E, 'weighted_E_miss005_2.csv');

%% Lorenz
clc, clf, clear, close all
tic
E_list = textread('20nodes.txt');
E_list = E_list(3:length(E_list),:);
E_num = length(E_list);
E_list = E_list+1;
A = zeros(20);
for i = 1:E_num
    A(E_list(i,2),E_list(i,1))=1;
end

num_edges = sum(sum(A~=0));
p = 0.25;
id = randperm(size(E_list,1));
E_list(id(1:floor(p*num_edges)),:) = [];
Anew = zeros(length(A));
for i = 1:size(E_list,1)
    Anew(E_list(i,2),E_list(i,1)) = 1;
end
writematrix(Anew, 'unweighted_adj_miss025_1.csv');
%% HR
clc, clf, clear, close all
A = csvread('unweighted_adj_20nodes.csv');
E_3 = csvread('edge_type_10I.csv');

E = [];
ii = 1;
for i = 1:length(A)
    for j = 1:length(A)
        if A(i,j) ~= 0
            E(ii,1) = j;
            E(ii,2) = i;
            %E(ii,3) = A(i,j);
            ii = ii+1;
        end
    end
end

E = [E,E_3];

num_edges = sum(sum(A~=0));
p = 0.1;
id = randperm(size(E,1));
E(id(1:floor(p*num_edges)),:) = [];
Anew = zeros(length(A));
for i = 1:size(E,1)
    Anew(E(i,2),E(i,1)) = E(i,3);
end
writematrix(Anew, 'EI_adj_miss010_2.csv');
writematrix(E(:,3), 'EI_E_miss010_2.csv');
