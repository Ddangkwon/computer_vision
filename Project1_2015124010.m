
%Data Read Part
img = imread('C:/Users/dram109/Desktop/Image/lena_grey.bmp');
imshow(img);
title('Ground Truth');
[height,width, channel] = size(img);
%Noise Generate part

%Impulse Noise(salt&pepper noise)

prob= 0.1;
imp_img = img;
for y = 1 : height
    for x = 1 : width
        rand_num = rand;
        if rand_num < prob
            if rand_num > prob / 2
                imp_img(x,y) = 255;%highligted by red color
            end
            if rand_num <= prob / 2 
                imp_img(x,y) = 0;
            end                     
        end
    end
end
figure(2);
imshow(imp_img);
title('impulse noise image');
sigma = 40;
gaussian_img = uint8(double(img) + randn(height, width) * sigma);
figure(3);
imshow(gaussian_img);
title('guassian noise image');

u_bound = 20;
l_bound = -20;
uniform_img = uint8(double(img) + (randn(height, width) - 0.5) * 50);
figure(4);
imshow(uniform_img);
title('uniform noise image');

temp_img = gaussian_img;

filter_size = 3;
sig = 5;
filter_img = zeros(width, height);
%res = uint8(conv2(img, filtering, 'same'));

for j = 1 : height
    for i = 1 : width
        [filtering, fil_img] = gaussian_filtering(temp_img, i, j, filter_size, sig);
        
        for y = 1:filter_size*2+1
            for x = 1:filter_size*2+1
              
                filter_img(j, i) = filter_img(j, i) + fil_img(y, x)*filtering(y,x);
                
            end
        end
     
    end
end
filter_img = uint8(filter_img);
figure(5);
imshow(filter_img);

title('Gaussian Filtered Image');

gaussian_psnr_noise_img = func_psnr(img, gaussian_img)
gaussian_psnr_filtered_img = func_psnr(img, filter_img)

filter_size = 5;
sig_d = 25;
sig_p = 100;
filter_img = zeros(width, height);
temp_img = uniform_img;
for j = 1 : height
    for i = 1 : width
        [filtering, fil_img] = bilateral_filtering(temp_img, i, j, filter_size, sig_d, sig_p);
        
        for y = 1:filter_size*2+1
            for x = 1:filter_size*2+1
              
                filter_img(j, i) = filter_img(j, i) + fil_img(y, x)*filtering(y,x);
                
            end
        end
     
    end
end
filter_img = uint8(filter_img);
figure(6);
imshow(filter_img);
title('Bilateral Filtered Image');
bilinear_psnr_noise_img = func_psnr(img, uniform_img)
bilinear_psnr_filtered_img = func_psnr(img, filter_img)


%median filter part
filt_size = 3*3;
med_kernel = zeros(filt_size,1);
% noise 이미지 불러오기
temp_img = imp_img;
filter_img = zeros(width, height);
%커널 사이즈 3*3이므로 한칸씩 여유공간을 확보한다.
for y = 2 : height -1
    for x = 2 : width-1
        med_kernel(1) = temp_img(y-1,x-1);
        med_kernel(2) = temp_img(y-1,x);
        med_kernel(3) = temp_img(y,x-1);
        med_kernel(4) = temp_img(y,x);
        med_kernel(5) = temp_img(y+1,x-1);
        med_kernel(6) = temp_img(y-1,x+1);
        med_kernel(7) = temp_img(y,x+1);
        med_kernel(8) = temp_img(y+1,x);
        med_kernel(9) = temp_img(y+1,x+1);
        mid = median(med_kernel);
        filter_img(y,x) = mid; 
    end
end
filter_img = uint8(filter_img);
figure(7);
imshow(filter_img);
title('Median Filtered Image');
median_psnr_noise_img = func_psnr(img,imp_img)
median_psnr_filtered_img = func_psnr(img, filter_img)
%bilateral filter 함수
function [filtered, filtered_img] = bilateral_filtering(img, i, j, size, sigma_d, sigma_p)
%out = zeros(size*2+1, size*2+1);
temp = 0;%sum값 저장해둘 임시 변수
 
filtered = zeros(size * 2 + 1, size * 2 +1);
filtered_img = zeros(size * 2 + 1, size* 2 +1);
for y = 1 : size*2+1
    for x = 1 : size*2+1
        if((x+i-size-1 > 0 && x+i-size-1 <= 512) && (y+j-size-1 > 0 && y+j-size-1 <= 512))
            %인접 픽셀간의 상관관계 정보 추가
            rel = abs(double(img(y,x)) - double(img(y+j-size-1, x+i-size-1))).^2;
            mag = (x-size-1)^2 + (y-size-1)^2;
            filtered(y,x) = ((exp(-mag/(2*(sigma_d^2)))) / sqrt(2 * pi * (sigma_d^2))).* ((exp(-rel/(2*(sigma_p^2)))) / sqrt(2 * pi * (sigma_p^2))) ;
            filtered_img(y,x) = img(y+j-size-1, x+i-size-1);
      
        end
        temp = temp + filtered(y,x);
    end
end
%normalization
filtered = filtered / temp;
filtered_img = uint8(filtered_img);
end

%%%%Guassian Filter
function [filtered, filtered_img] = gaussian_filtering(img, i, j, size, sigma)
%out = zeros(size*2+1, size*2+1);
temp = 0;%sum값 저장해둘 임시 변수
filtered = zeros(size * 2 + 1, size * 2 +1);
filtered_img = zeros(size * 2 + 1, size* 2 +1);
for y = 1 : size*2+1
    for x = 1 : size*2+1
        if((x+i-size-1 > 0 && x+i-size-1 <= 512) && (y+j-size-1 > 0 && y+j-size-1 <= 512))
          
            mag = (x-size-1)^2 + (y-size-1)^2;
            filtered(y,x) = (exp(-mag/(2*(sigma^2)))) / (2 * pi * (sigma^2));
            filtered_img(y,x) = img(y+j-size-1, x+i-size-1);
        end
        temp = temp + filtered(y,x);
    end
end
%normalization
filtered = filtered / temp;
filtered_img = uint8(filtered_img);
end



function psnr = func_psnr(gt, fil_img)
[height, width, ch] = size(gt);
gt = double(gt);
fil_img = double(fil_img);
sqt = abs(gt - fil_img).^2;
mse = sum(sqt(:)) / (height * width * ch);%2차원 1차원으로 flatten
psnr = 10*log10((255^2) / mse);

end
