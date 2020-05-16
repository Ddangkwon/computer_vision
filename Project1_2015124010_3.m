
clear all;
%Data Read Part
close all;
img = imread('C:/Users/dram109/Desktop/Image/lena.bmp');
imshow(img);
title('Ground Truth');
[width, height, channel] = size(img);

%Bayer pattern images

bayer_img = zeros(width, height,channel);
r_img = zeros(width, height,channel);
rs_img = zeros(width, height,channel);
%a)
%[B,G
% G,R];
%[2,2] 배열필터에서 (0,0) 은 B값을 , (0,1), (1,0)위치는 G값을, (1,1)위치에 R값을 대입한다.
for h = 1 : height
    for w= 1: width
        %나눗셈 퍼센트 연산(나머지) 수행
        if (mod(h, 2) == 0 && mod(w, 2) == 0)
      % Pick red value.
            bayer_img(w, h,1) = img(w, h, 1);
        elseif (mod(h, 2) == 0 && mod(w, 2) == 1) || (mod(h, 2) == 1 && mod(w, 2) == 0)
      % Pick green value
            bayer_img(w, h,2) = img(w, h, 2);
        elseif (mod(h, 2) == 1 && mod(w, 2) == 1)
      % Pick blue value.
            bayer_img(w, h,3) = img(w, h, 3);
        end
    end
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%b)Demosaicked image part
redChannel = bayer_img(:, :, 1);
greenChannel = bayer_img(:, :, 2);
blueChannel = bayer_img(:, :, 3);
%(redChannel +greenChannel +blueChannel);
figure;
imshow(uint8(bayer_img));
inter_img = zeros(width, height, channel);
%Mosaicked image를 먼저 넣는다.
tic
inter_img(:, :, 1) =  redChannel;
inter_img(:, :, 2) = greenChannel;
inter_img(:, :, 3) = blueChannel;
    for h = 1 : height
        for w= 1: width
            %1) case 511 * 511 내부 정사각형 부분에 대한 bilinear interpoltion
            if(h > 1 &&  h < height && w > 1 && w < width)
                %R값만 존재 G,B사용해서 보간
                if(mod(h,2) == 0 && mod(w,2) == 0)
                        count = 0;
                        sum = 0;
                        if(blueChannel(h-1, w+1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h-1, w+1);
                        end
                        if(blueChannel(h+1, w-1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h+1, w-1);
                        end
                        if(blueChannel(h-1, w-1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h-1, w-1);
                        end
                        if (blueChannel(h+1, w+1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h+1, w+1);
                        end
                       add = sum / count;
                       inter_img(h, w, 3) = add;
                       count = 0;
                       sum = 0;
                        if(greenChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h, w+1);
                        end
                        if(greenChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum +greenChannel(h, w-1);
                        end
                        if(greenChannel(h-1, w)~=0)
                            count = count + 1;
                            sum =sum + greenChannel(h-1, w);
                        end
                        if (greenChannel(h+1, w)~=0)
                            count = count + 1;
                            sum =sum + greenChannel(h+1, w);
                        end
                       add = sum / count;
                       %greenChannel(h, w) = sum;
                       inter_img(h, w, 2) = add;
                %B값만 존재 G,R사용해서 보간
                elseif(mod(h,2) == 1 && mod(w,2) == 1)
                        count = 0;
                        sum = 0;
                      
                        if(redChannel(h-1, w-1)~=0)
                            count = count + 1;
                            sum = sum + redChannel(h-1, w-1);
                        end
                        if(redChannel(h+1, w+1)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h+1, w+1);
                        end
                        if(redChannel(h-1, w+1)~=0)
                            count = count + 1;
                            sum =sum + redChannel(h-1, w+1);
                        end
                        if (redChannel(h+1, w-1)~=0)
                            count = count + 1;
                            sum =sum + redChannel(h+1, w-1);
                        end
                  
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            inter_img(h, w, 1) = add; 
                       
                       
                       count = 0;
                        sum = 0;
                        if(greenChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h, w+1);
                        end
                        if(greenChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum +greenChannel(h, w-1);
                        end
                        if(greenChannel(h-1, w)~=0)
                            count = count + 1;
                            sum =sum + greenChannel(h-1, w);
                        end
                        if (greenChannel(h+1, w)~=0)
                            count = count + 1;
                            sum =sum + greenChannel(h+1, w);
                        end
                       add = sum / count;
                 
                       inter_img(h, w, 2) = add;
                %G값 R,B사용해서 보간
                else
                    
                    
                       inter_img(h, w, 2) = add;
                        count = 0;
                        sum = 0;
                        if(blueChannel(h-1, w)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h-1, w);
                        end
                        if(blueChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h, w-1);
                        end
                        if(blueChannel(h+1, w)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h+1, w);
                        end
                        if (blueChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h, w+1);
                        end
                        add = sum / count;
                        inter_img(h, w, 3) = add;
                        count = 0;
                        sum = 0;
                     
                        if(redChannel(h-1, w)~=0)
                            count = count + 1;
                            sum = sum + redChannel(h-1, w);
                        end
                        if(redChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h, w-1);
                        end
                        if(redChannel(h, w+1)~=0)
                            count = count + 1;
                            sum =sum + redChannel(h, w+1);
                        end
                        if (redChannel(h+1, w)~=0)
                            count = count + 1;
                            sum =sum + redChannel(h+1, w);
                        end
                  
                        add = sum / count;
                        %redChannel(h, w) = sum;
                        inter_img(h, w, 1) = add;         
                       
                       
                end
            %이제 사각형 모서리 부분의 공백을 채운다. 
            %현재 B or R로만 채워진 case와 G로 채워진 이 두 case에 대해 처리한다.
            else
                %B에 대한 보간 Part
                if(mod(h,2) == 1 && mod(w,2) == 1 && (h ==1 && w ~= 1))
                        count = 0;
                        sum = 0;
                      
       
                        if(redChannel(h+1, w+1)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h+1, w+1);
                        end
                        
                  
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            inter_img(h, w, 1) = add; 
                       
                       
                       count = 0;
                        sum = 0;
                        if(greenChannel(h+1, w)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h+1, w);
                        end
                        if(greenChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h, w+1);
                        end
                        if (greenChannel(h, w-1)~=0)
                            count = count + 1;
                            sum =sum + greenChannel(h, w-1);
                        end
                       add = sum / count;
                       %greenChannel(h, w) = sum;
                       inter_img(h, w, 2) = add;
                elseif(mod(h,2) == 1 && mod(w,2) == 1 && (w ==1 && h ~= 1))
                        count = 0;
                        sum = 0;
                      
       
                        if(redChannel(h+1, w+1)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h+1, w+1);
                        end
                        
                  
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            inter_img(h, w, 1) = add; 
                       
                       
                       count = 0;
                        sum = 0;
                        if(greenChannel(h+1, w)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h+1, w);
                        end
                        if(greenChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h, w+1);
                        end
                        if (greenChannel(h-1, w)~=0)
                            count = count + 1;
                            sum =sum + greenChannel(h-1, w);
                        end
                       add = sum / count;
                       %greenChannel(h, w) = sum;
                       inter_img(h, w, 2) = add;
                %width가 더 클때 case
                
                %width가 더 클때 case
                elseif(mod(h,2) == 1 && mod(w,2) == 1 && (h ==1 && w == 1))
                        count = 0;
                        sum = 0;
                      
       
                       if(redChannel(h+1, w+1)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h+1, w+1);
                        end
                        
                  
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            inter_img(h, w, 1) = add; 
                       
                       
                       count = 0;
                        sum = 0;
                        
                        if(greenChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h, w+1);
                        end
                        if (greenChannel(h+1, w)~=0)
                            count = count + 1;
                            sum =sum + greenChannel(h+1, w);
                        end
                       add = sum / count;
                       %greenChannel(h, w) = sum;
                       inter_img(h, w, 2) = add;
                %R에 대한 보간 Part
                elseif(mod(h,2) == 0 && mod(w,2) == 0 && w == width && h ~= height)
                    count = 0;
                        sum = 0;
                      
       
                        if(blueChannel(h-1, w-1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h-1, w-1);
                        end
                        
                  
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            inter_img(h, w, 3) = add; 
                       
                       
                       count = 0;
                        sum = 0;
                        if(greenChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h, w-1);
                        end
                        if(greenChannel(h-1, w)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h-1, w);
                        end
                        if (greenChannel(h+1, w)~=0)
                            count = count + 1;
                            sum =sum + greenChannel(h+1, w);
                        end
                       add = sum / count;
                       %greenChannel(h, w) = sum;
                       inter_img(h, w, 2) = add;
                elseif(mod(h,2) == 0 && mod(w,2) == 0 && h == height && w ~= width)
                    count = 0;
                        sum = 0;
                      
       
                        if(blueChannel(h-1, w-1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h-1, w-1);
                        end
                        
                  
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            inter_img(h, w, 3) = add; 
                       
                       
                       count = 0;
                        sum = 0;
                        if(greenChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h, w-1);
                        end
                        if(greenChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h, w+1);
                        end
                        if (greenChannel(h-1, w)~=0)
                            count = count + 1;
                            sum =sum + greenChannel(h-1, w);
                        end
                       add = sum / count;
                       %greenChannel(h, w) = sum;
                       inter_img(h, w, 2) = add;
                elseif(mod(h,2) == 0 && mod(w,2) == 0 && h == height && w == width)
                    count = 0;
                        sum = 0;
                      
       
                        if(blueChannel(h-1, w-1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h-1, w-1);
                        end
                        
                  
                        add = sum / count;
                        %redChannel(h, w) = sum;
                        inter_img(h, w, 3) = add; 
                       
                       
                       count = 0;
                        sum = 0;
                        if(greenChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h, w-1);
                        end
                        
                        if (greenChannel(h-1, w)~=0)
                            count = count + 1;
                            sum =sum + greenChannel(h-1, w);
                        end
                       add = sum / count;
                       %greenChannel(h, w) = sum;
                       inter_img(h, w, 2) = add;
                end
                %G에 대한 보간 Part
                if((mod(h,2) == 1 && mod(w,2) == 0)  && h == 1 && w ~= width)
                        count = 0;
                        sum = 0;
                      
       
                        if(redChannel(h+1, w)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h+1, w);
                        end
                        
                  
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            inter_img(h, w, 1) = add; 
                       
                       count = 0;
                        sum = 0;
                       
                        if(blueChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h, w-1);
                        end
                       
                        if (blueChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h, w+1);
                        end
                        add = sum / count;
                        inter_img(h, w, 3) = add;
                elseif((mod(h,2) == 1 && mod(w,2) == 0)  && h ~= 1 && w == width)
                        count = 0;
                        sum = 0;
                      
       
                        if(redChannel(h+1, w)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h+1, w);
                        end
                        if(redChannel(h-1, w)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h-1, w);
                        end
                        
                  
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            inter_img(h, w, 1) = add; 
                       
                       count = 0;
                        sum = 0;
                       
                        if(blueChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h, w-1);
                        end
                        add = sum / count;
                        inter_img(h, w, 3) = add;
                %우상단 모서리 G값 보간
                elseif((mod(h,2) == 1 && mod(w,2) == 0)  && h == 1 && w == width)
                        count = 0;
                        sum = 0;

                        if(redChannel(h+1, w)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h+1, w);
                        end
                
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            inter_img(h, w, 1) = add; 
                       
                       count = 0;
                        sum = 0;
                        if(blueChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h, w-1);
                        end
                        add = sum / count;
                        inter_img(h, w, 3) = add;      
                %나머지 좌측 G값 보간
                elseif((mod(h,2) == 0 && mod(w,2) == 1)  && w == 1 && h ~= height)
                    count = 0;
                        sum = 0;
                      
       
                        if(redChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h, w+1);
                        end
                        add = sum / count;
                        %redChannel(h, w) = sum;
                        inter_img(h, w, 1) = add; 
                  
                         
                       if(blueChannel(h+1, w)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h+1, w);
                        end
                       
                        if (blueChannel(h-1, w)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h-1, w);
                        end
                        add = sum / count;
                        inter_img(h, w, 3) = add;
                %하단 G값 보간
                elseif((mod(h,2) == 0 && mod(w,2) == 1)  && w ~= 1 && h == height)
                        count = 0;
                        sum = 0;
                      
       
                        if(redChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h, w+1);
                        end
                        if(redChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h, w-1);
                        end
                        add = sum / count;
                        %redChannel(h, w) = sum;
                        inter_img(h, w, 1) = add; 
                  
                           
                       if(blueChannel(h-1, w)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h-1, w);
                       end
                       add = sum / count;
                       inter_img(h, w, 3) = add;
                 %좌하단 G값 보간
                 elseif((mod(h,2) == 0 && mod(w,2) == 1)  && w == 1 && h == height)
                        count = 0;
                        sum = 0;
                       if(redChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h, w+1);
                        end
                
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            inter_img(h, w, 1) = add; 
                       
                       count = 0;
                        sum = 0;
                        if(blueChannel(h-1, w)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h-1, w);
                        end
                        add = sum / count;
                        inter_img(h, w, 3) = add;
                        
              
                end
            end
        end
    end
toc    
            
%c) PSNR Check Part
figure;
imshow(uint8(inter_img));
check_psnr = func_psnr(img, inter_img)
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%d) mean, variance check part
hist_arr_R = zeros(256,1);
hist_arr_G = zeros(256,1); 
hist_arr_B = zeros(256,1); 
hist_arr_DR = zeros(256,1);
hist_arr_DG = zeros(256,1); 
hist_arr_DB = zeros(256,1);


for y = 1: height
    for x = 1: width
        pixel_intensity_R = img(y,x,1)+1;
        pixel_intensity_G = img(y,x,2)+1;
        pixel_intensity_B = img(y,x,3)+1;
        hist_arr_R(pixel_intensity_R) = hist_arr_R(pixel_intensity_R) + 1;
        hist_arr_G(pixel_intensity_G) = hist_arr_G(pixel_intensity_G) + 1;
        hist_arr_B(pixel_intensity_B) = hist_arr_B(pixel_intensity_B) + 1;
     
    end
end
for y = 1: height
    for x = 1: width
        pixel_intensity_R = uint8(inter_img(y,x,1))+1;
        pixel_intensity_G = uint8(inter_img(y,x,2))+1;
        pixel_intensity_B = uint8(inter_img(y,x,3))+1;
        hist_arr_DR(pixel_intensity_R) = hist_arr_DR(pixel_intensity_R) + 1;
        hist_arr_DG(pixel_intensity_G) = hist_arr_DG(pixel_intensity_G) + 1;
        hist_arr_DB(pixel_intensity_B) = hist_arr_DB(pixel_intensity_B) + 1;
    end
end
%Ground Truth mean
mean_R = mean(mean(img(:, :, 1)))
mean_G = mean(mean(img(:, :, 2)))
mean_B = mean(mean(img(:, :, 3)))
%Demosaicked Image mean
mean_DR = mean(mean(inter_img(:, :, 1)))
mean_DG = nanmean(mean(inter_img(:, :, 2)))
mean_DB = nanmean(mean(inter_img(:, :, 3)))
%Ground Truth variance
var_R = var(var(double(img(:, :, 1))))
var_G = var(var(double(img(:, :, 2))))
var_B = var(var(double(img(:, :, 3))))
%Ground Truth variance
var_DR = var(var(double(inter_img(:, :, 1))))
var_DG = var(var(double(inter_img(:, :, 2))))
var_DB = var(var(double(inter_img(:, :, 3))))
figure;
plot(hist_arr_R);
hold on
plot(hist_arr_DR, 'r');
title('Image Histogram_R');
legend({'Ground Truth','Demosaicked Image'},'Location','southwest')
figure;
plot(hist_arr_G);
hold on
plot(hist_arr_DG,'r');
title('Image Histogram_G');
legend({'Ground Truth','Demosaicked Image'},'Location','southwest')
figure;
plot(hist_arr_B);
hold on
plot(hist_arr_DB, 'r');
title('Image Histogram_B');
legend({'Ground Truth','Demosaicked Image'},'Location','southwest')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%)e Build your own algorithm to improve the visual or quantitative quality
%현재 Bilinear Interpolation 결과의 경우 인간이 구별하기 어려워지는 30db이상의 성능을 확보했기 때문에
%Computational Complexity문제를 극복하고자 했습니다.
%이미지의 특성상 인접 픽셀끼리의 R,G,B값은 서로 거의 비슷하다는 특징에서 아이디어를 떠올렸습니다
%R,G,B모두 인접 두개 픽셀만을 조사해서 그 차가 정해진 Threshold값 이상일 경우 두 픽셀의 mean값을 이하일 경우 해당
%픽셀에서 가까운 값 또는 거리가 동일한 경우 random한 하나의 값을 고르는 방식으로 진행합니다
own_img = zeros(width, height, channel);
threshold = 100;
tic
own_img(:, :, 1) =  redChannel;
own_img(:, :, 2) = greenChannel;
own_img(:, :, 3) = blueChannel;
    for h = 1 : height
        for w= 1: width
            if(h > 1 &&  h < height && w > 1 && w < width)
                %R값만 존재 G,B사용해서 보간
                if(mod(h,2) == 0 && mod(w,2) == 0)
                        count = 0;
                        sum = 0;    
                        if(blueChannel(h-1, w-1)~=0 && blueChannel(h+1, w+1)~=0)
                            if(blueChannel(h+1, w+1) - blueChannel(h-1, w-1) > threshold)
                                count = count + 2;
                                sum = sum + blueChannel(h-1, w-1);
                                sum = sum + blueChannel(h+1, w+1);
                            else
                                count = count + 1;
                                sum = sum + blueChannel(h+1, w+1);
                            end  
                        end
                       
                       add = sum / count;
                       own_img(h, w, 3) = add;
                       count = 0;
                       sum = 0;
                       if(greenChannel(h, w-1)~=0 && greenChannel(h, w+1)~=0)
                            if(greenChannel(h, w+1) - greenChannel(h, w-1) > threshold)
                                count = count + 2;
                                sum = sum + greenChannel(h, w-1);
                                sum = sum + greenChannel(h, w+1);
                            else
                                count = count + 1;
                                sum = sum + greenChannel(h, w-1);
                            end  
                        end
                        
                       add = sum / count;
          
                       own_img(h, w, 2) = add;
               %B값만 존재할 때 R,G값을 사용해 보간
               elseif(mod(h,2) == 1 && mod(w,2) == 1)
                        count = 0;
                        sum = 0;    
                        if(redChannel(h-1, w-1)~=0 && redChannel(h+1, w+1)~=0)
                            if(redChannel(h+1, w+1) - redChannel(h-1, w-1) > threshold)
                                count = count + 2;
                                sum = sum + redChannel(h-1, w-1);
                                sum = sum + redChannel(h+1, w+1);
                                
                            else
                                count = count + 1;
                                sum = sum + redChannel(h+1, w+1);
                            end  
                        end
                       add = sum / count;
                       
                       own_img(h, w, 1) = add;
                       count = 0;
                       sum = 0;
                       if(greenChannel(h, w-1)~=0 && greenChannel(h, w+1)~=0)
                            if(greenChannel(h, w+1) - greenChannel(h, w-1) > threshold)
                                count = count + 2;
                                sum = sum + greenChannel(h, w-1);
                                sum = sum + greenChannel(h, w+1);
                            else
                                count = count + 1;
                                sum = sum + greenChannel(h, w-1);
                            end  
                        end
                        
                       add = sum / count;
              
                       own_img(h, w, 2) = add;
                %G값만 존재할 때 R,B값을 사용해 보간
                elseif((mod(h,2) == 0 && mod(w,2) == 1))
                    count = 0;
                    sum = 0;
                    if(redChannel(h, w-1)~=0 && redChannel(h, w+1)~=0)
                            if(redChannel(h, w+1) - redChannel(h, w-1) > threshold)
                                count = count + 2;
                                sum = sum + redChannel(h, w-1);
                                sum = sum + redChannel(h, w+1);
                            else
                                count = count + 1;
                                sum = sum + redChannel(h, w+1);
                            end  
                     end
                       
                     add = sum / count;
                     own_img(h, w, 1) = add;
                     count = 0;
                       sum = 0;
                       if(blueChannel(h-1, w)~=0 && blueChannel(h+1, w)~=0)
                            if(blueChannel(h+1, w) - blueChannel(h-1, w) > threshold)
                                count = count + 2;
                                sum = sum + blueChannel(h-1, w);
                                sum = sum + blueChannel(h+1, w);
                            else
                                count = count + 1;
                                sum = sum + blueChannel(h+1, w);
                            end  
                        end
                        
                       add = sum / count;
                       count = 0;
                       sum = 0;
                       own_img(h, w, 3) = add;
                elseif((mod(h,2) == 1 && mod(w,2) == 0))
                    count = 0;
                    sum = 0;
                    if(redChannel(h+1, w)~=0 && redChannel(h-1, w)~=0)
                            if(redChannel(h+1, w) - redChannel(h-1, w) > threshold)
                                count = count + 2;
                                sum = sum + redChannel(h+1, w);
                                sum = sum + redChannel(h-1, w);
                            else
                                count = count + 1;
                                sum = sum + redChannel(h+1, w);
                            end  
                     end
                       
                     add = sum / count;
                     own_img(h, w, 1) = add;
                     count = 0;
                       sum = 0;
                       if(blueChannel(h, w-1)~=0 && blueChannel(h, w+1)~=0)
                            if(blueChannel(h, w+1) - blueChannel(h, w-1) > threshold)
                                count = count + 2;
                                sum = sum + blueChannel(h, w+1);
                                sum = sum + blueChannel(h, w-1);
                            else
                                count = count + 1;
                                sum = sum + blueChannel(h, w+1);
                            end  
                        end
                        
                       add = sum / count;
            
                       own_img(h, w, 3) = add;
                
                end 
            else
             
                %B에 대한 보간 Part
                if(mod(h,2) == 1 && mod(w,2) == 1 && (h ==1 && w ~= 1))
                        count = 0;
                        sum = 0;
                      
       
                        if(redChannel(h+1, w+1)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h+1, w+1);
                        end
                        
                  
                            add = sum / count;
                         
                            own_img(h, w, 1) = add; 
                       
                       
                       count = 0;
                        sum = 0;
                        if(greenChannel(h+1, w)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h+1, w);
                        end
                        if(greenChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h, w+1);
                        end
                        if (greenChannel(h, w-1)~=0)
                            count = count + 1;
                            sum =sum + greenChannel(h, w-1);
                        end
                       add = sum / count;
                       %greenChannel(h, w) = sum;
                       own_img(h, w, 2) = add;
                elseif(mod(h,2) == 1 && mod(w,2) == 1 && (w ==1 && h ~= 1))
                        count = 0;
                        sum = 0;
                      
       
                        if(redChannel(h+1, w+1)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h+1, w+1);
                        end
                        
                  
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            own_img(h, w, 1) = add; 
                       
                       
                       count = 0;
                        sum = 0;
                        if(greenChannel(h+1, w)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h+1, w);
                        end
                        if(greenChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h, w+1);
                        end
                        if (greenChannel(h-1, w)~=0)
                            count = count + 1;
                            sum =sum + greenChannel(h-1, w);
                        end
                       add = sum / count;
                       %greenChannel(h, w) = sum;
                       own_img(h, w, 2) = add;
                %width가 더 클때 case
                
                %width가 더 클때 case
                elseif(mod(h,2) == 1 && mod(w,2) == 1 && (h ==1 && w == 1))
                        count = 0;
                        sum = 0;
                      
       
                       if(redChannel(h+1, w+1)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h+1, w+1);
                        end
                        
                  
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            own_img(h, w, 1) = add; 
                       
                       
                       count = 0;
                        sum = 0;
                        
                        if(greenChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h, w+1);
                        end
                        if (greenChannel(h+1, w)~=0)
                            count = count + 1;
                            sum =sum + greenChannel(h+1, w);
                        end
                       add = sum / count;
                       %greenChannel(h, w) = sum;
                       own_img(h, w, 2) = add;
                %R에 대한 보간 Part
                elseif(mod(h,2) == 0 && mod(w,2) == 0 && w == width && h ~= height)
                    count = 0;
                        sum = 0;
                      
       
                        if(blueChannel(h-1, w-1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h-1, w-1);
                        end
                        
                  
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            own_img(h, w, 3) = add; 
                       
                       
                       count = 0;
                        sum = 0;
                        if(greenChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h, w-1);
                        end
                        if(greenChannel(h-1, w)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h-1, w);
                        end
                        if (greenChannel(h+1, w)~=0)
                            count = count + 1;
                            sum =sum + greenChannel(h+1, w);
                        end
                       add = sum / count;
                       %greenChannel(h, w) = sum;
                       own_img(h, w, 2) = add;
                elseif(mod(h,2) == 0 && mod(w,2) == 0 && h == height && w ~= width)
                    count = 0;
                        sum = 0;
                      
       
                        if(blueChannel(h-1, w-1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h-1, w-1);
                        end
                        
                  
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            own_img(h, w, 3) = add; 
                       
                       
                       count = 0;
                        sum = 0;
                        if(greenChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h, w-1);
                        end
                        if(greenChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h, w+1);
                        end
                        if (greenChannel(h-1, w)~=0)
                            count = count + 1;
                            sum =sum + greenChannel(h-1, w);
                        end
                       add = sum / count;
                       %greenChannel(h, w) = sum;
                       own_img(h, w, 2) = add;
                elseif(mod(h,2) == 0 && mod(w,2) == 0 && h == height && w == width)
                    count = 0;
                        sum = 0;
                      
       
                        if(blueChannel(h-1, w-1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h-1, w-1);
                        end
                        
                  
                        add = sum / count;
                        %redChannel(h, w) = sum;
                        own_img(h, w, 3) = add; 
                       
                       
                       count = 0;
                        sum = 0;
                        if(greenChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum + greenChannel(h, w-1);
                        end
                        
                        if (greenChannel(h-1, w)~=0)
                            count = count + 1;
                            sum =sum + greenChannel(h-1, w);
                        end
                       add = sum / count;
                       %greenChannel(h, w) = sum;
                       own_img(h, w, 2) = add;
                end
                %G에 대한 보간 Part
                if((mod(h,2) == 1 && mod(w,2) == 0)  && h == 1 && w ~= width)
                        count = 0;
                        sum = 0;
                      
       
                        if(redChannel(h+1, w)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h+1, w);
                        end
                        
                  
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            own_img(h, w, 1) = add; 
                       
                       count = 0;
                        sum = 0;
                       
                        if(blueChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h, w-1);
                        end
                       
                        if (blueChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h, w+1);
                        end
                        add = sum / count;
                        own_img(h, w, 3) = add;
                elseif((mod(h,2) == 1 && mod(w,2) == 0)  && h ~= 1 && w == width)
                        count = 0;
                        sum = 0;
                      
       
                        if(redChannel(h+1, w)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h+1, w);
                        end
                        if(redChannel(h-1, w)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h-1, w);
                        end
                        
                  
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            own_img(h, w, 1) = add; 
                       
                       count = 0;
                        sum = 0;
                       
                        if(blueChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h, w-1);
                        end
                        add = sum / count;
                        own_img(h, w, 3) = add;
                %우상단 모서리 G값 보간
                elseif((mod(h,2) == 1 && mod(w,2) == 0)  && h == 1 && w == width)
                        count = 0;
                        sum = 0;

                        if(redChannel(h+1, w)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h+1, w);
                        end
                
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            own_img(h, w, 1) = add; 
                       
                       count = 0;
                        sum = 0;
                        if(blueChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h, w-1);
                        end
                        add = sum / count;
                        own_img(h, w, 3) = add;      
                %나머지 좌측 G값 보간
                elseif((mod(h,2) == 0 && mod(w,2) == 1)  && w == 1 && h ~= height)
                    count = 0;
                        sum = 0;
                      
       
                        if(redChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h, w+1);
                        end
                        add = sum / count;
                        %redChannel(h, w) = sum;
                        own_img(h, w, 1) = add; 
                  
                         
                       if(blueChannel(h+1, w)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h+1, w);
                        end
                       
                        if (blueChannel(h-1, w)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h-1, w);
                        end
                        add = sum / count;
                        own_img(h, w, 3) = add;
                %하단 G값 보간
                elseif((mod(h,2) == 0 && mod(w,2) == 1)  && w ~= 1 && h == height)
                        count = 0;
                        sum = 0;
                      
       
                        if(redChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h, w+1);
                        end
                        if(redChannel(h, w-1)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h, w-1);
                        end
                        add = sum / count;
                        %redChannel(h, w) = sum;
                        own_img(h, w, 1) = add; 
                  
                           
                       if(blueChannel(h-1, w)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h-1, w);
                       end
                       add = sum / count;
                       own_img(h, w, 3) = add;
                 %좌하단 G값 보간
                 elseif((mod(h,2) == 0 && mod(w,2) == 1)  && w == 1 && h == height)
                        count = 0;
                        sum = 0;
                       if(redChannel(h, w+1)~=0)
                            count = count + 1;
                            sum = sum +redChannel(h, w+1);
                        end
                
                            add = sum / count;
                            %redChannel(h, w) = sum;
                            own_img(h, w, 1) = add; 
                       
                       count = 0;
                        sum = 0;
                        if(blueChannel(h-1, w)~=0)
                            count = count + 1;
                            sum = sum + blueChannel(h-1, w);
                        end
                        add = sum / count;
                        own_img(h, w, 3) = add;
                        
              
                end
               
            end

        end    
    end
toc
figure;
imshow(uint8(own_img));
check_psnr = func_psnr(img, own_img)    
            
function psnr = func_psnr(gt, fil_img)
[height, width, ch] = size(gt);
gt = double(gt);
fil_img = double(fil_img);
sqt = abs(gt - fil_img).^2;
mse = nansum(sqt(:)) / (height * width * ch)%2차원 1차원으로 flatten
psnr = 10*log10((255^2) / mse);

end