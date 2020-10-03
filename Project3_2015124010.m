
clear all
close all

point = rand(15000,3);
k_point = rand(600,3);
figure()
nearest_p = zeros(600,3);
size(point)
size(k_point)
%2D Plot�� ���� 2���� ǥ��

scatter3(point(:,1),point(:,2),point(:,3),'LineWidth',2)       % Plot cluster centers
figure
scatter3(k_point(:,1),k_point(:,2),k_point(:,3),'LineWidth',2)       % Plot cluster centers
size(k_point(:,2))
size(point(:,2))
hold on
tic 
 for i = 1 : 600
     dist = 0;
     idx  = 0;
     tmp_x = 0;
     tmp_y = 0;
     tmp_z = 0;
     min = 9999;
     for j = 1 : 10000
            dist = sqrt((k_point(i,1) - point(j,1)).^2 + (k_point(i,2) - point(j,2)).^2 + (k_point(i,3) - point(j,3)).^2);
            if dist < min
                min = dist
                idx = i;
                tmp_x = point(j,1);
                tmp_y = point(j,2);
                tmp_z = point(j,3);
            end

     end
     nearest_p(i,1) = tmp_x;
     nearest_p(i,2) = tmp_y;
     nearest_p(i,3) = tmp_z;
     
 end
 %figure
 toc
 scatter3(nearest_p(:,1),nearest_p(:,2),nearest_p(:,3),'LineWidth',2)       % Plot cluster centers
 % Sample point cloud

tic
%KD-Tree ����
tree_Q = Create_KDtree(point)

% Sample points
figure
scatter3(k_point(:,1),k_point(:,2),k_point(:,3),'LineWidth',2)    
 hold on
% Compute the nearest neighbours
for i = 1:size(k_point(:,2))
    
    p_nn = KDtreeSearch(tree_Q, k_point(i,:));    
    scatter3(p_nn(1), p_nn(2), p_nn(3),'ro','LineWidth',2)   
    hold on
end
toc
 %kd-tree implementation
%P�� KD-Tree �Է� point cloud data
%direction�� �� �����ǹ�
function [ tree ] = Create_KDtree( point, direction )


tree = Struct_KDtree;

%nargin=>�Է� �μ��� ����
%x�� ���� ����
if nargin < 2
    direction = 1;
end
tree.direction = direction;


%x��=>y��=>z�� �������� tree 
p_direction = point(:, tree.direction);
if tree.direction == 1
    nextDirection = 2;
elseif tree.direction == 2
    nextDirection = 3; 
else        
    nextDirection = 1;
end


tree.thres = mean(p_direction);

%Leafnode�� �̸� ������ ��������� ����
% case: right
point_right = point(p_direction<tree.thres, :);
[N, ~] = size(point_right);
%leaf��尡 �ƴҽ�
if N >= 2
    tree.right = Create_KDtree(point_right, nextDirection);
else
    tree.right = point_right;
end

% case: left
point_left = point(p_direction>=tree.thres, :);
[N, ~] = size(point_left);
if N >= 2
    tree.left = Create_KDtree(point_left, nextDirection);
else
    tree.left = point_left;
end

end

%������ KD-Tree ������� nearest neighbor search ����
%�Է°� tree�� ��� KD-Tree
%p�� �ֱ��� �̿��� ã�����ϴ� point �Է°�
function [ p_nn ] = KDtreeSearch( tree, p )
%�� �������� ���� ���� ���� Ȯ���ϰ� ũ�� ������ ������ �������� �������� search�� ����
%��������� 
if (p(tree.direction) < tree.thres)
    [~, N] = size(tree.right);
    %N>=2�̻��̸� �ش� ��忡�� �ٽ� ��������� Search����
    if N == 1
        p_nn = KDtreeSearch(tree.right, p);     
    %�ֱ��� �̿� �߰�
    else
        p_nn = tree.right;
    end
else
    [~, N] = size(tree.left);
    if N == 1
        p_nn = KDtreeSearch(tree.left, p);
    else
        p_nn = tree.left;
    end
end
end