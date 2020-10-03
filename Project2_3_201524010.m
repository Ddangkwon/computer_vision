
close all
im = (imread('C:/Users/dram109/Desktop/codeAndfile_2015124010/Image/starbucks_image.png'));                    % Load Image
%ima = imresize(im, [300,500]);
ima =im;
%Step I. Resize query image into 32×32
logo = (imread('C:/Users/dram109/Desktop/codeAndfile_2015124010/Image/logo.png'));  


[H,W,c] = size(ima);
figure;
imshow(ima);
figure;
imshow(logo);
logo =rgb2gray(logo);
logo_img=double(logo);
size(logo)
img=rgb2gray(ima);
%img = im2double(ima);%code for clustering
min = 9999;
%feat최소값을 담을 벡터
min_feat = zeros(100,1);
minx = 0;
miny = 0;

size(img);
tic
feat = local_feat_vec(logo);
size(feat)
%%%%%%%%%%Target image processing%%%%%%%%%%%%
%Step V. Divide target image into overlapped 32×32 image.
%Step VI. For every 32×32 image, follow the same processes in Step II ~ IV.
%Step VII. Check whether each 32×32 image is well matched to the query or not
result=[];
for y = 1 : 2 : H 
    for x = 1: 2 : W

        if(y>0 && y < H - 32 && x > 0 && x < W-32)
           crop_img = zeros(32,32);
           for j = y : y +32
               for i = x : x+32
                   
                   crop_img( j-y+1 ,i-x+1) =  img(j,i);
               end
           end
           temp = local_feat_vec(crop_img);
            
           result =sumabs(feat - temp);

           if(min>result)
              min = result
              minx = x;
              miny = y;
  

           end
            
        end
            
    end
    
end
minx
miny
figure;
imshow(ima);
hold on
h = images.roi.Rectangle(gca,'Position',[minx miny 32 32],'StripeColor','r');

toc
%%%%%%%%% own algorithm part%%%%%%%%%%%%%
threshold =70;
result_c=0;
feat_c = revised_local_feat_vec(logo);
figure;
imshow(ima);
tic
for y = 1 : 2 : H 
    for x = 1: 2 : W

        if(y>0 && y < H - 32 && x > 0 && x < W-32)
           crop_img_c = zeros(32,32);
           sum = 0;
           suma = 0;
           for j = y : y +32
               for i = x : x+32
              
                   crop_img_c( j-y+1 ,i-x+1) =  img(j,i);
                   
               end
           end
           c_img = double(crop_img_c);
           for j = y+14 : y +21
               for i = x+14 : x+21
                   suma = suma + abs(logo_img(j-y+1 ,i-x+1) - double(crop_img_c(j-y+1 ,i-x+1)));
               end
           end
           suman = double(suma / (8 * 8));
           temp_c = revised_local_feat_vec(crop_img_c);

            result_c = sumabs(feat_c - temp_c);
             
           
           %result_c
           
           if(5.5 > result_c && suman < threshold)
              min = result_c
              minx = x;
              miny = y;
              
              hold on
              h = images.roi.Rectangle(gca,'Position',[minx miny 32 32],'StripeColor','r');

           end
            
        end
            
    end
    
end
minx
miny
toc

%%%%Query image processing%%%%%%
function feat = local_feat_vec(im)

rows=size(im,1);
cols=size(im,2);
ch = size(im,3);
if ch==3
    im=rgb2gray(im);
end
im=double(im);


%Step II. Find its gradient images, ??, ??.
dx=im; 
dy=im; 

% gradient 계산
for i=1:rows-2
    dy(i,:)=(im(i,:)-im(i+2,:));
end
for i=1:cols-2
    dx(:,i)=(im(:,i)-im(:,i+2));
end
 % Matrix containing the angles of each edge gradient
angle=atand(dx./dy);
angle=imadd(angle,180); %Angles in range (0,180)
magnitude=sqrt(dx.^2 + dy.^2);

% nan성분 제거 nan성분으로 인해 에러가 발생
angle(isnan(angle))=0;
magnitude(isnan(magnitude))=0;
feat=[]; %initialized the feature vector

for i = 0: rows/8 - 2
    for j= 0: cols/8 -2
      
        
        mag_block = magnitude(8*i+1 : 8*i+16 , 8*j+1 : 8*j+16);
        ang_block = angle(8*i+1 : 8*i+16 , 8*j+1 : 8*j+16);
        
        b_feat=[];
        %Step III. Design the oriented histogram for each 8× 8 cell, just like a part of SIFT.

        for x= 0:1
            for y= 0:1
                angleA =ang_block(8*x+1:8*x+8, 8*y+1:8*y+8);
                magA   =mag_block(8*x+1:8*x+8, 8*y+1:8*y+8); 
                %1*8벡터 선언
                histr  =zeros(1,8);
                
      
                for p=1:8
                    for q=1:8
%                       
                        alpha= angleA(p,q);
                        %magnitude와 angle에 관한 oreinted histogram작성(8방향)
                        if alpha>0 && alpha<=45
                            histr(1)=histr(1)+ magA(p,q)*(45-alpha);
                
                        elseif alpha>45 && alpha<=90
                            histr(2)=histr(2)+ magA(p,q)*(90-alpha);                 
      
                        elseif alpha>90 && alpha<=135
                            histr(3)=histr(3)+ magA(p,q)*(135-alpha);
       
                        elseif alpha>135 && alpha<=180
                            histr(4)=histr(4)+ magA(p,q)*(180-alpha);
    
                        elseif alpha>180 && alpha<=225
                            histr(5)=histr(5)+ magA(p,q)*(225-alpha);
             
                        elseif alpha>225 && alpha<=270
                            histr(6)=histr(6)+ magA(p,q)*(270-alpha);
 
                        elseif alpha>270 && alpha<=315
                            histr(7)=histr(7)+ magA(p,q)*(315-alpha);
  
                        elseif alpha>315 && alpha<=360
                            histr(8)=histr(8)+ magA(p,q)*(360-alpha);
        
                 
                        end
                    end
                end
              
                %Step IV. Finally, build a feature vector for 32× 32 image. (It will be 4×4×8=128-dim)
                b_feat=[b_feat histr]; % Concatenation of Four histograms to form one block feature
        
                                
            end
        end
      
        feat=[feat b_feat]; %Features concatenation
        
    end
end


% error방지를 위해 nan성분 제거 및 Normalization
feat(isnan(feat))=0; 
feat=feat/sqrt(norm(feat)^2+.001);
%feat vector의 값들 중 터무니 없는 값들을 제어할 필요성이 있어 normalized 결과를 확인하고 특정 threshold값 이상을 잘라주었다. 
for z=1:length(feat)
    if feat(z)>0.1
         feat(z)=0;
    end
end
feat=feat/sqrt(norm(feat)^2+.001);        
    
end



function r_feat = revised_local_feat_vec(im)
rows=size(im,1);
cols=size(im,2);
ch = size(im,3);
if ch==3
    im=rgb2gray(im);
end
im=double(im);


%Step II. Find its gradient images, ??, ??.
dx=im; %Basic Matrix assignment
dy=im; %Basic Matrix assignment

% gradient 계산
for i=1:rows-2
    dy(i,:)=(im(i,:)-im(i+2,:));
end
for i=1:cols-2
    dx(:,i)=(im(:,i)-im(:,i+2));
end

angle=atand(dx./dy); % Matrix containing the angles of each edge gradient
angle=imadd(angle,180); %Angles in range (0,180)
magnitude=sqrt(dx.^2 + dy.^2);

% nan성분 제거 nan성분으로 인해 에러가 발생
angle(isnan(angle))=0;
magnitude(isnan(magnitude))=0;
r_feat=[]; %initialized the feature vector
% Iterations for Blocks
for i = 0: rows/8 - 2
    for j= 0: cols/8 -2
      
        
        mag_block = magnitude(8*i+1 : 8*i+16 , 8*j+1 : 8*j+16);
        ang_block = angle(8*i+1 : 8*i+16 , 8*j+1 : 8*j+16);
        
        b_feat=[];
        %Step III. Design the oriented histogram for each 8× 8 cell, just like a part of SIFT.
    
        for x= 0:1
            for y= 0:1
                angleA =ang_block(8*x+1:8*x+8, 8*y+1:8*y+8);
                magA   =mag_block(8*x+1:8*x+8, 8*y+1:8*y+8); 
                %1*8벡터 선언
                histr  =zeros(1,6);
                
          
                for p=1:8
                    for q=1:8
%                       
                        alpha= angleA(p,q);
                        %magnitude와 angle에 관한 oreinted histogram작성(8방향)
                        if alpha>0 && alpha<=60
                            histr(1)=histr(1)+ magA(p,q)*(60-alpha);
                
                        elseif alpha>60 && alpha<=120
                            histr(2)=histr(2)+ magA(p,q)*(120-alpha);                 
      
                        elseif alpha>120 && alpha<=180
                            histr(3)=histr(3)+ magA(p,q)*(180-alpha);
       
                        elseif alpha>180 && alpha<=240
                            histr(4)=histr(4)+ magA(p,q)*(240-alpha);
    
                        elseif alpha>240 && alpha<=300
                            histr(5)=histr(5)+ magA(p,q)*(300-alpha);
             
                        elseif alpha>300 && alpha<=360
                            histr(6)=histr(6)+ magA(p,q)*(360-alpha);

        
                 
                        end
                    end
                end
              
                %Step IV. Finally, build a feature vector for 32× 32 image. (It will be 4×4×8=128-dim)
                b_feat=[b_feat histr]; % Concatenation of Four histograms to form one block feature
        
                                
            end
        end
        r_feat=[r_feat b_feat]; %Features concatenation
        
    end
end


% error방지를 위해 nan성분 제거 및 Normalization
r_feat(isnan(r_feat))=0; 
r_feat=r_feat/sqrt(norm(r_feat)^2+.001);
%feat vector의 값들 중 터무니 없는 값들을 제어할 필요성이 있어 normalized 결과를 확인하고 특정 threshold값 이상을 잘라주었다. 
for z=1:length(r_feat)
    if r_feat(z)>0.1
         r_feat(z)=0;
    end
end
r_feat=r_feat/sqrt(norm(r_feat)^2+.001);        
   
end


