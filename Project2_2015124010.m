clc
clear all
close all
tic
im = (imread('C:/Users/dram109/Desktop/codeAndfile_2015124010/Image/house.bmp'));                    % Load Image
%Image크기 resize를 통해 대표값을 뽑아 Clustering을 plot하는 방식을 ㄷ
ima = imresize(im, [128,128]);
imshow(ima);
img = im2double(ima);%code for clustering
%img = im2double(im);%code for check image
[H,W,ch] = size(img);
Flatten = reshape(img,H*W,3);                 % Color Features
H
W
size(Flatten);
K     = 2;                                            % number of segments
centeroids = Flatten( ceil(rand(K,1)*size(Flatten,1)) ,:);             % Cluster Centers
dis_labels   = zeros(size(Flatten,1),K+2);                   
KMI   = 10;         
color_arr    = '+rxg*boc+m+rogobocomo';   
tic
for n = 1:KMI
   for i = 1:size(Flatten,1)
      for j = 1:K  
        dis_labels(i,j) = norm(Flatten(i,:) - centeroids(j,:));      
      end
      [dist,label] = min(dis_labels(i,1:K));              
      dis_labels(i,K+1) = label;                               % Lable값을 저장하는 arr
      dis_labels(i,K+2) = dist;                          % 최소 거리를 저장하는 arr
   end
   for i = 1:K
      A = (dis_labels(:,K+1) == i);                          % Cluster K Points
      centeroids(i,:) = mean(Flatten(A,:));                      % New Cluster Centers
      if sum(isnan(centeroids(:))) ~= 0                    % nan성분 있을 경우 error발생해서 없는 부분만 탐지 될 수 있도록한다.
         NC = find(isnan(centeroids(:,1)) == 1);           % Find Nan Centers
         for Ind = 1:size(NC,1)
         centeroids(NC(Ind),:) = Flatten(randi(size(Flatten,1)),:);
         end
      end
   end
end

X = zeros(size(Flatten));

for i = 1:K
    idx = find(dis_labels(:,K+1) == i);
    X(idx,:) = repmat(centeroids(i,:),size(idx,1),1); 
end
T = reshape(X,H,W,3);
figure()
imshow(img);
figure()
imshow(T); 
toc;
disp('Segment K='); disp(K)

figure()
view(3),axis vis3d, rotate3d on
hold on
%2D Plot을 위해 2차원 표현
 for i = 1:K
point = Flatten(dis_labels(:,K+1) == i,:)*255.0;                           % Find points value of each cluster    

scatter3(point(:,1),point(:,2),point(:,3),color_arr(2*i-1:2*i),'LineWidth',2)       % Plot cluster centers
scatter3(centeroids(:,1)*255.0,centeroids(:,2)*255.0,centeroids(:,3)*255.0,'*k','LineWidth',7);       % Plot cluster centers

 end
 hold off
 
xlabel('R'), ylabel('G'), zlabel('B')
%RGB to YUV conversion
figure()
hold on
view(3),axis vis3d, rotate3d on
 for i = 1:K
pt = Flatten(dis_labels(:,K+1) == i,:)*255.0;                           % Find points of each cluster
Y = 0.299 * pt(:,1) + 0.587 * pt(:,2) + 0.114 * pt(:,3);
U = 0.492*(pt(:,3) );
V = 0.877*(pt(:,1) );
scatter3(Y(:),U(:),V(:),'LineWidth',2);    % Plot points with determined color and shape

 end
 hold off

xlabel('Y'), ylabel('U'), zlabel('V')

 %대표값이 아닌 전체 이미지에 대한 YUV to RGB변환
temp_img = T;
Y_ch = 0.299 * temp_img(:, :, 1)+ 0.587 * temp_img(:, :, 2) + 0.114 * temp_img(:, :, 3);
U_ch = 0.492*((temp_img(:, :, 3)) - Y_ch);
V_ch = 0.877*((temp_img(:, :, 1)) - Y_ch);
R_ch = Y_ch + 1.140*V_ch;
G_ch = Y_ch - 0.395*U_ch - 0.581*V_ch;
B_ch = Y_ch + 2.032*U_ch;
conv_img = zeros(H,W,3);
size(conv_img(:, :, 1))
size(R_ch)
conv_img(:, :, 1) = R_ch;
conv_img(:, :, 2) = G_ch;
conv_img(:, :, 3) = B_ch;

figure()
imshow((conv_img));
title('YUV to RGB Conversion');

%mean shift
%mean shift의 경우 과정이 K-means에 비해 복잡하게 진행됨에 따라 함수로 선언
arr_y=size(zeros(100, 1));
cent= 0;
arr_n=size(zeros(100, 1));
K = 0;
%이미지를 2차원으로 Flatten
flt = reshape(img,H*W,3); 
size(flt)
h = 3 *3;
k = 1;
y = zeros(size(flt));
%gaussian 분포의 random 값으로 이미지에 적용하기 전에 test용으로 사용 
arr_rnd = normrnd(3,10,[100,1]);
size(arr_rnd)
h = 2;
%result =  mean_shift(flt, y, h ,k)
[arr_y, cent, arr_n, K] = meanshift_clustering(flt ,h);
size(arr_y)
size(cent)
size(arr_n)
size(K)
output = resize(arr_y,H,W,3);
figure
imshow(ima); title('input image');
figure
imshow(output); title('meanshift segmented image')
figure
view(3),axis vis3d, rotate3d on
scatter3(parr_n(:,1),arr_n(:,2),arr_n(:,3),'LineWidth',2); 



xlabel('R'), ylabel('G'), zlabel('B')
%[arr_y, arr_v, cent, arr_n, K] = meanshift_clustering_rgb(arr_x, m, h, k)

function f=gaussian_filter(n,s,Y,X)
x = -1/2:1/(n-1):1/2;
[Y,X] = meshgrid(x,x);
f = exp( -(X.^2+Y.^2)/(2*s^2) );
f = f / sum(f(:));
end

function res = kernel_gaussian(y,x,h)

for i = 1: 2*h+1

        dis = abs(y-x);
        
        if(dis <=1)
            res = exp(dis.^2);
        else
            res = 0;
        end
 
end

end

function res = kernel_flat(y,x,h)

for i = 1: 2*h+1
 
        dis = abs(y-x);
        
        if(dis <=1)
            res = 1;
        else
            res = 0;
        end

end

end



function res = mean_shift(arr, y, h, kinds)
%h=kernel size
%k = kernel type
len = length(arr);
denom = 0;%분모항 변수 선언
nume = 0;%분자항 변수 선언
for i = 1:len
    val = arr(i,:);
    if(abs(val-y) >= h)
        med = 0;
    else
        if(kinds==0)
            med = kernel_flat(y,val,h);%fla kernel
        else
            med = kernel_gaussian(y,val,h);%gaussian kernel
            
        end
        
    end
    denom = denom + med;
    nume = nume + val .* med;   
end
res = nume / denom;
end

function [res,h] = meanshift_n(arr_x,y,n)
len = length(arr_x);
nw = zeros(len,1);
arr_z = [nw, arr_x];

for i = 1: len
    x = arr_x(i,:);
    arr_z(i,1) = norm(y-x);
end
%해당 열 기준으로 행을 오름차순 정렬해서 높은 값부터 정렬 수행
arr_z = sortrows(arr_z,1);

Trans_arr = arr_z(1:n,:);

h = Trans_arr(n,1);
Trans_arr(:,1) = exp(-(Trans_arr(:,1)/h).^2);
nume = sum (Trans_arr(:,1).*[Trans_arr(:,1),Trans_arr(:,2),Trans_arr(:,3)]); % 
denom = sum(Trans_arr(:,1));
if(denom == 0)
    res = y;
else
    res = nume /denom;
end
%sz_check = size(y)

end
%clustering 수행
%erode dilate방식으로 재귀적으로 군집을 설정
function arr_y = recursive_clustered(arr, arr_y, i, h)
iter = 0;
%반복 횟수를 정해서 그 이상일 경우 break로 무한 loop를 방지
for j = 1:length(arr)
    if((arr_y(j) == 0) && (norm(arr(j,:) - arr(i,:)) <= h) && iter < 10)
        arr_y(j) = arr_y(i);
        arr_y = recursive_clustered(arr, arr_y, i, h);
        iter = iter + 1;
    end
    
end
end
%arr_y 군집 label, cent 군집 중심, arr_n 군집의 샘플 개수, K 군집 개수
function [arr_y, cent, arr_n, K]  = clustering(arr_v,h)
len = length(arr_v);
K = 0;
arr_y = zeros(1,1);
%해당 arr_y가 비어있지 않다면 값이 들어있고 따라서 군집의 개수는 1 증가하고 군집화를 반복한다.
for i = 1:len
    if(arr_y(i) ~= 0)
        K = K + 1;
        arr_y(i) = K;
        arr_y = recursive_clustered(arr_v, arr_y, i, h);
    end
    
end
cent = zeros(K,1);
arr_n = zeros(K,1);

%군집 내의 샘플 개수를 세는 과정과 cent값을 할당하느 과정
for i = 1:K 
    for j = 1:len
        if(arr_y(j) == i)
            arr_n(i) = arr_n(i) + 1;
            cent(i) = cent(i) + arr_v(j);
        end
    end

end
cent = cent./arr_n;
end

function [arr_y, cent, arr_n, K] = meanshift_clustering(arr_x,h)
arr_v = zeros(size(arr_x));

for i = 1 : length(arr_x)   
    y = arr_x(i,:);
    size(y)
    hsum = 0;
    cnt = 0;
    while(true)
        [yy,hh] =  meanshift_n(arr_x, y, h);
        %size(yy)
        %size(y)
        hsum = hsum + hh;
        cnt = cnt + 1;
        if(norm(yy - y) <= 0.2)
            break;
        end
        y = yy;
    end
  
    arr_v(i,:) = yy;
    h = hsum / cnt;
end
[arr_y, cent, arr_n, K] = clustering(arr_v,h);

end
