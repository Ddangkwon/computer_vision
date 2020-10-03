close all
img_l = (imread('surf_image/surf_image.png'));                   
img_r = (imread('surf_image/surf_image_l.png')); 
size(img_r)
img_l = rgb2gray(img_l);
img_r = rgb2gray(img_r);

l_points = detectSURFFeatures(img_l);
[l_features,l_valid_points] = extractFeatures(img_l,l_points);
figure; imshow(img_l); hold on;
plot(l_valid_points.selectStrongest(10),'showOrientation',true);
%feature point x, y좌표값
l_p = uint8(l_valid_points.selectStrongest(10).Location)
r_points = detectSURFFeatures(img_r);
[r_features,r_valid_points] = extractFeatures(img_r,r_points);
figure; imshow(img_r); hold on;
plot(r_valid_points.selectStrongest(10),'showOrientation',true);
%feature point x, y좌표값
r_p = uint32(r_valid_points.selectStrongest(10).Location)

match_plot(img_l,img_r, l_p, r_p)
title('full search')
lmse(img_l,img_r, l_p, r_p)
title('lmse')
ransac(img_l,img_r, l_p, r_p)
title('ransac')


%full search 방식의 feature matching
function h = match_plot(img1,img2,points1,points2)

h = figure;
colormap = {'b','r','m','y','g','c','r','g','b','c'};
height = max(size(img1,1),size(img2,1));

points2 = [points2(:,2) points2(:,1)];
match_img = zeros(height, size(img1,2)+size(img2,2), size(img2,3));
match_img(1:size(img1,1),1:size(img1,2),:) = img1;
match_img(1:size(img2,1),size(img1,2)+1:end,:) = img2;
imshow(uint8(match_img));
size(match_img)
hold on;
pix_l=0;
pix_r=0;

for i=1:size(points1,1)
    pix_l = match_img(points1(i,2),points1(i,1));

    plot(points1(i,1),points1(i,2),'x')
    temp = 0;
    min = 999;
    for j=1:size(points2,1)

        pix_r = match_img(points2(j,1),points2(j,2)+size(img1,2));
        plot(points2(j,2)+size(img1,2),points2(j,1),'o')
        temp = (pix_l- pix_r)^2;
        if(temp < min)
            min = temp
            l_x = points1(i,1);
            l_y = points1(i,2);
            r_x = points2(j,2)+size(img1,2)
            r_y =  points2(j,1)
        end
         plot([r_x l_x],[r_y l_y],colormap{mod(i,10)+1}); 
    end
  
    
end
hold off;
end

function res = lmse(img1,img2,points1,points2)


h = figure;
colormap = {'b','r','m','y','g','c','r','g','b','c'};
height = max(size(img1,1),size(img2,1));

points2 = [points2(:,2) points2(:,1)];
match_img = zeros(height, size(img1,2)+size(img2,2), size(img2,3));
match_img(1:size(img1,1),1:size(img1,2),:) = img1;
match_img(1:size(img2,1),size(img1,2)+1:end,:) = img2;
imshow(uint8(match_img));
size(match_img)
hold on;
pix_l=0;
pix_r=0;
N = size(points1(:,1))
length(N(2))
A = [];
B = [];

%각 Feature point들을 바탕으로 A,B 행렬을 구한다.
for i = 1 : 2* N
    temp = [];
    if mod(i,2) == 0
        temp = [points1(round(i/2),1),points1(round(i/2),2),1 , 0, 0, 0];
        A = vertcat(A,temp);
    else
        temp = [0,0,0,points1(round(i/2),1),points1(round(i/2),2),1 ];
        A = vertcat(A,temp);
    end
end


for i = 1 : 2* N
    temp = [];
    if mod(i,2) == 0
        temp = points2(round(i/2),1);
        B = vertcat(B,temp);
    else
        temp = points2(round(i/2),2);
        B = vertcat(B,temp);
    end
    
end
size(B)
size(A)
A = double(A);
B = double(B);
tx = [];
%값을 오차함수의 편미분 값이 0되는 t값을 찾고 DoF=6인 T행렬를 찾는다.
tx = inv(A'*A)*A'*B
size(tx)
temp = [0,0,1];
T = vertcat([tx(1),tx(2),tx(3)],[tx(4),tx(5),tx(6)], temp)
size(T);
T = double(T);
hold on;
pix_l=0;
pix_r=0;

for i=1:size(points1,1)
    pix_l = match_img(points1(i,2),points1(i,1));

    plot(points1(i,1),points1(i,2),'x')
    temp = 0;
    min = 999;
    result = [];
    po = double([points1(i,1);
          points1(i,2);
          1]);
    result = T * po;
    size(result);
    

            l_x = points1(i,1);
            l_y = points1(i,2);
            r_x = uint32(result(2,1)+size(img1,2));
            r_y =  uint32(result(1,1));
       
         plot([r_x l_x],[r_y l_y],colormap{mod(i,10)+1}); 
    
  
    
end
hold off;

end


function res = ransac(img1,img2,points1,points2)


h = figure;
colormap = {'b','r','m','y','g','c','r','g','b','c'};
height = max(size(img1,1),size(img2,1));

points2 = [points2(:,2) points2(:,1)];
match_img = zeros(height, size(img1,2)+size(img2,2), size(img2,3));
match_img(1:size(img1,1),1:size(img1,2),:) = img1;
match_img(1:size(img2,1),size(img1,2)+1:end,:) = img2;
imshow(uint8(match_img));
size(match_img)
hold on;
pix_l=0;
pix_r=0;
N = size(points1(:,1))
length(N(2))
A = [];
B = [];
%1~10사이의 랜덤한 난수를 생성한다.(3점
r = round(10*rand(3,1))
iter= 10;


%Random하게 뽑은 세 쌍의 feature point를 바탕으로 A,B 행렬을 구한 후 T를 추정한다.
for x = 1 : iter
    
for i = 1 : 2*size(r)
    temp = [];
    if mod(i,2) == 0
        temp = [points1(r(round(i/2)),1),points1(r(round(i/2)),2),1 , 0, 0, 0];
        A = vertcat(A,temp);
    else
        temp = [0,0,0,points1(r(round(i/2)),1),points1(r(round(i/2)),2),1 ];
        A = vertcat(A,temp);
    end
end


for i = 1 : 2*size(r)
    temp = [];
    if mod(i,2) == 0
        temp = points2(r(round(i/2),1),1);
        B = vertcat(B,temp);
    else
        temp = points2(round(i/2),2);
        B = vertcat(B,temp);
    end
    
end

end
size(B)
size(A)
A = double(A);
B = double(B);
tx = [];
%값을 오차함수의 편미분 값이 0되는 t값을 찾고 DoF=6인 T행렬를 찾는다.
tx = inv(A'*A)*A'*B
size(tx)
temp = [0,0,1];
T = vertcat([tx(1),tx(2),tx(3)],[tx(4),tx(5),tx(6)], temp)
size(T);
T = double(T);
hold on;
pix_l=0;
pix_r=0;


for i=1:size(points1,1)
    pix_l = match_img(points1(i,2),points1(i,1));

    plot(points1(i,1),points1(i,2),'x')
    temp = 0;
    min = 999;
    result = [];
    po = double([points1(i,1);
          points1(i,2);
          1]);
    result = T * po;
    size(result);
    

            l_x = points1(i,1);
            l_y = points1(i,2);
            r_x = uint32(result(2,1)+size(img1,2));
            r_y =  uint32(result(1,1));
       
         plot([r_x l_x],[r_y l_y],colormap{mod(i,10)+1}); 
    
  
    
end
hold off;

end