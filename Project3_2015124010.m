
clear all
close all

point = rand(15000,3);
k_point = rand(600,3);
figure()
nearest_p = zeros(600,3);
size(point)
size(k_point)
%2D Plot을 위해 2차원 표현

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
%KD-Tree 생성
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
%P는 KD-Tree 입력 point cloud data
%direction은 축 방향의미
function [ tree ] = Create_KDtree( point, direction )


tree = Struct_KDtree;

%nargin=>입력 인수의 개수
%x축 부터 시작
if nargin < 2
    direction = 1;
end
tree.direction = direction;


%x축=>y축=>z축 방향으로 tree 
p_direction = point(:, tree.direction);
if tree.direction == 1
    nextDirection = 2;
elseif tree.direction == 2
    nextDirection = 3; 
else        
    nextDirection = 1;
end


tree.thres = mean(p_direction);

%Leafnode에 이를 때까지 재귀적으로 수행
% case: right
point_right = point(p_direction<tree.thres, :);
[N, ~] = size(point_right);
%leaf노드가 아닐시
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

%생성한 KD-Tree 기반으로 nearest neighbor search 진행
%입력값 tree의 경우 KD-Tree
%p는 최근접 이웃을 찾고자하는 point 입력값
function [ p_nn ] = KDtreeSearch( tree, p )
%각 차원축의 분할 기준 값을 확인하고 크면 오른쪽 작으면 왼쪽으로 내려가며 search를 진행
%재귀적으로 
if (p(tree.direction) < tree.thres)
    [~, N] = size(tree.right);
    %N>=2이상이면 해당 노드에서 다시 재귀적으로 Search수행
    if N == 1
        p_nn = KDtreeSearch(tree.right, p);     
    %최근접 이웃 발견
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