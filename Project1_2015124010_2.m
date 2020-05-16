%Data Read Part
close all;
img = imread('C:/Users/dram109/Desktop/Image/lena_grey.bmp');
c_img = imread('C:/Users/dram109/Desktop/Image/lena.bmp');
imshow(img);
figure;
imshow(c_img);
title('Ground Truth');
%height, width���� Lena Gray�� ����Ǿ� ����.
[height,width, channel] = size(img);
redChannel = c_img(:, :, 1);
greenChannel = c_img(:, :, 2);
blueChannel = c_img(:, :, 3);
%pixel���� ������ ���� array�� �����Ѵ�.
hist_arr = zeros(256,1);

hist_arr_eq = zeros(256,1); 
hist_eq = zeros(256,1);
cdf = zeros(256,1);
%��ü �˰���� hist �迭
hist_arr_eq_own = zeros(256,1);
%RGB Equalization��
hist_arr_R = zeros(256,1); 
hist_arr_G = zeros(256,1); 
hist_arr_B = zeros(256,1);
hist_arr_eq_R = zeros(256,1); 
hist_arr_eq_G = zeros(256,1); 
hist_arr_eq_B = zeros(256,1);

hist_eq_R = zeros(256,1); 
hist_eq_G = zeros(256,1); 
hist_eq_B = zeros(256,1); 

cdf_R = zeros(256,1);
cdf_G = zeros(256,1);
cdf_B = zeros(256,1);
cdf_eq = zeros(256,1);
cdf_eq_R = zeros(256,1);
cdf_eq_G = zeros(256,1);
cdf_eq_B = zeros(256,1);
accum = 0;
%RGB�� ���� �̹��� ������׷� ��Ȱȭ �����
rgb_out = zeros(height, width, 3);
%gray�� ���� �̹��� ������׷�

for y = 1: height
    for x = 1: width
        pixel_intensity = img(y,x);
        accum = accum + double(img(y,x));
        hist_arr(pixel_intensity) = hist_arr(pixel_intensity) + 1;
    end
end
accum_ori = accum;

figure;
plot(hist_arr);
title('Image Histogram');
figure;
bar(hist_arr);
title('Image Histogram(bar)');
sum = 0;
total = 0;
max_hist = 0;


%CDF�� ���ϱ� ���� ��ü ���� ���� �����ش�
for x = 1: 255
    sum = sum + hist_arr(x);
    cdf(x) = sum/(height*width);
end

figure;
plot(cdf);
sum = 0;
%CDF�� 8��Ʈ �ȼ����� �ִ밪�� �����ְ� round�� ���� linear�ϰ� �����.
for x = 1: 255
    hist_eq(x) = round(cdf(x) * (255));
end

    
%����� CDF�� �̹����� �����Ͽ� Histogram Equalization ������ �̹����� �����Ѵ�. 
out_img = hist_eq(img) + 1;
accum = 0;
for y = 1: height
    for x = 1: width
        pixel_intensity_2 = out_img(y,x);
        accum = accum + double(out_img(y,x));
        hist_arr_eq(pixel_intensity_2) = hist_arr_eq(pixel_intensity_2)+1;
    end
end
figure;
bar(hist_arr_eq);
%HE�� �̹����� CDF�� Ȯ���� ����. 
for x = 1: 255
    sum = sum + hist_arr_eq(x);
    cdf_eq(x) = sum/(height*width);
end
figure;
plot(cdf_eq,'r');



figure;
imshow(uint8(out_img));
title('Equalized gray Image');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%C)�ڽŸ��� �˰���
min_pixel = 0;
max_pixel = 0;
for i = 1 : 255
    %�ȼ��� ���� �߿� ���� ���� �� ��ġ
    if(i <= 100 && hist_arr(i)==0)
      min_pixel  = i;
    end
    %�ȼ��� ���� �߿� ���� ���� �� ��ġ  hist_arr(i+1) == 0�� ������ġ
    %�̹��� Ư���� Ư������ �������� 0�� Ȯ���� �ſ� ����
    if(i > 100 && hist_arr(i)==0 && hist_arr(i+1) == 0)
      max_pixel  = i;
      break;
    end
end
accum = 0;
prob = (max_pixel - min_pixel) / 255;
mid = (max_pixel - min_pixel) / 2 ;
for y = 1: height
    for x = 1: width
        pixel_intensity_3 = img(y,x);
        accum = accum + double(pixel_intensity_3);
        hist_arr_eq_own(pixel_intensity_3) =  hist_arr_eq_own(pixel_intensity_3) + prob;
      
    end
end
temp_h = max_pixel;
temp_l = min_pixel;
check_l = (mid - temp_l) / temp_l;
check_h = (temp_h - mid) / (255-temp_h);
for i = 1 : 255
    %�ȼ��� ���� �߿� ���� ���� �� ��ġ
    
    sum = 0;
    if(i <= mid )
        num = 0;
        tmp_i = temp_l+ i;
        while num < check_l
            sum = sum + hist_arr_eq_own(tmp_i);
            num = num + 1;
            tmp_i = tmp_i + 1; 
        end
      hist_arr_eq_own(i) = hist_arr_eq_own(i)+(sum / num) * (1-prob);
    end
    %�ȼ��� ���� �߿� ���� ���� �� ��ġ  hist_arr(i+1) == 0�� ������ġ
    %�̹��� Ư���� Ư������ �������� 0�� Ȯ���� �ſ� ����
    if(i > mid+5)
      num = 0;
        tmp_i = mid + 255 - i;
        while num < check_h
            sum = sum + hist_arr_eq_own(tmp_i);
            num = num + 1;
            tmp_i = tmp_i + 1; 
        end
      hist_arr_eq_own(i) = hist_arr_eq_own(i)+(sum / num) * (1-prob);
    end
end
figure;
bar(hist_arr_eq_own);
out_img_own = hist_arr_eq_own(img) + 1;
uint8(out_img_own);
figure;
imshow(uint8(out_img_own));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%color�� ���� �̹��� ������׷�

for y = 1: height
    for x = 1: width
        pixel_intensity_R = c_img(y,x,1)+1;
        pixel_intensity_G = c_img(y,x,2)+1;
        pixel_intensity_B = c_img(y,x,3)+1;
        hist_arr_R(pixel_intensity_R) = hist_arr_R(pixel_intensity_R) + 1;
        hist_arr_G(pixel_intensity_G) = hist_arr_G(pixel_intensity_G) + 1;
        hist_arr_B(pixel_intensity_B) = hist_arr_B(pixel_intensity_B) + 1;
     
    end
end

figure;
bar(hist_arr_R);
title('Image Histogram_R');

figure;
bar(hist_arr_G);
title('Image Histogram_G');

figure;
bar(hist_arr_B);
title('Image Histogram_B');
sum_R=0;
sum_G=0;
sum_B=0;
for x = 1: 255
    sum_R = sum_R + hist_arr_R(x);
    sum_G = sum_G + hist_arr_G(x);
    sum_B = sum_B + hist_arr_B(x);
    cdf_R(x) = sum_R/(height*width);
    cdf_G(x) = sum_G/(height*width);
    cdf_B(x) = sum_B/(height*width);
end


figure;
plot(cdf_R,'r');
title('cdf_r');
figure;
plot(cdf_G,'g');
title('cdf_r');
figure;
plot(cdf_B,'b');
title('cdf_r');
sum = 0;
%CDF�� 8��Ʈ �ȼ����� �ִ밪�� �����ְ� round�� ���� linear�ϰ� �����.
for x = 1: 256
    hist_eq_R(x) = round(cdf_R(x) * (255));
    hist_eq_G(x) = round(cdf_G(x) * (255));
    hist_eq_B(x) = round(cdf_B(x) * (255));
end

    
%����� CDF�� �̹����� �����Ͽ� Histogram Equalization ������ �̹����� �����Ѵ�. 
rgb_out(:,:,1) = hist_eq_R(c_img(:,:,1)+1);
rgb_out(:,:,2) = hist_eq_G(c_img(:,:,2)+1);
rgb_out(:,:,3) = hist_eq_B(c_img(:,:,3)+1);
accum = 0;
for y = 1: height
    for x = 1: width
        pixel_intensity_R = rgb_out(y,x,1)+1;
        pixel_intensity_G = rgb_out(y,x,2)+1;
        pixel_intensity_B = rgb_out(y,x,3)+1;
        hist_arr_eq_R(pixel_intensity_R) = hist_arr_eq_R(pixel_intensity_R) + 1;
        hist_arr_eq_G(pixel_intensity_G) = hist_arr_eq_G(pixel_intensity_G) + 1;
        hist_arr_eq_B(pixel_intensity_B) = hist_arr_eq_B(pixel_intensity_B) + 1;
    end
end
figure;
bar(hist_arr_eq_R);
title('Image Histogram_Equalized_R');
figure;
bar(hist_arr_eq_G);
title('Image Histogram_Equalized_g');
figure;
bar(hist_arr_eq_B);
title('Image Histogram_Equalized_b');

sum_R=0;
sum_G=0;
sum_B=0;
for x = 1: 255
    sum_R = sum_R + hist_arr_eq_R(x);
    sum_G = sum_G + hist_arr_eq_G(x);
    sum_B = sum_B + hist_arr_eq_B(x);
    cdf_eq_R(x) = sum_R/(height*width);
    cdf_eq_G(x) = sum_G/(height*width);
    cdf_eq_B(x) = sum_B/(height*width);
end

figure;
plot(cdf_eq_R,'r');
title('cdf_r_equalized');
figure;
plot(cdf_eq_G,'g');
title('cdf_r_equalized');
figure;
plot(cdf_eq_B,'b');
title('cdf_r_equalized');



figure;
imshow(uint8(rgb_out));
title('Equalized Image');

