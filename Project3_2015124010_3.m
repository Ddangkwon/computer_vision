close all
colormap gray

train_dat = load('mnist_train.csv');
test_dat = load('mnist_test.csv');
train_imgs_data = train_dat(:,2:785);
test_imgs_data = test_dat(:,2:785);
train_dat_l = train_dat(:,1);
test_dat_l = test_dat(:,1);
size(train_dat)
size(test_dat)
%N은 TRAIN DATA개수, M은 테스트 데이터 개수
N = 600;
M = 1000;
train_labels = zeros(N, 1);
test_labels = zeros(M, 1);
train_imgs = zeros(N, 784);
test_imgs = zeros(M, 784);
%학습 데이터와 훈련 데이터 크기 조정
for i = 1 : N
train_labels(i) = train_dat_l(i,:);
train_imgs(i,:) = train_imgs_data(i,:);

end
for i = 1 : M
test_labels(i) = test_dat_l(i,:);
test_imgs(i,:) = test_imgs_data(i,:);
end
size(train_imgs)
%t=reshape(train_imgs(11,:),28,28);
imagesc(reshape(train_imgs(1,:),28,28)');

check_label_value = train_labels(1)


%전방 계산 테스트
%[cls, dat] = feed_foward_mnist((train_imgs(1,:)),u,v,alpha);

alpha = 0.1;
lr =  0.05;
%p의 경우 은닉층의 노드 개수로 칭한다.(inp
p =80;
r_cnt = 200;
%학습 데이터에 대한 학습 진행
[u,v,err] = mlp_training_mnist(train_imgs, train_labels, alpha, lr, p, r_cnt);


%테스트 데이터에 대해서 성능 평가 진행(전방 계산만)
[err_t] = test_model_mnist(test_imgs, test_labels, alpha, lr, p, r_cnt, u, v);
cnt = 1 : 1 : r_cnt;
%학습 결과 Plot
figure 
plot(cnt, err, '-or');
title('training result')
xlabel('epoch')
ylabel('error rate')
final_error_rate = err(r_cnt)
%테스트 성능 평가 결과


final_error_rate = err_t

acc = 1 - err_t



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%K-fold cross validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K= 6;
data_per_fold = N / K;
remainder =  N - (N / K);
train_k_labels = zeros(remainder, 1);
test_k_labels = zeros(data_per_fold, 1);
train_k_imgs = zeros(remainder, 784);
test_k_imgs = zeros(data_per_fold, 784);
err_rate = zeros(K,1);
total = 0;
for i = 1 : K
    for k = 1 : data_per_fold
        %테스트 데이터를 먼저 뽑는다.
        test_k_labels(k) = train_dat_l((i-1)*data_per_fold +k,:);
        test_k_imgs(k,:) = train_imgs_data((i-1)*data_per_fold +k,:);
    end
    temp = 0;
   for j = 1 : K
       %테스트 데이터로 뽑힌 데이터를 제외한 모든 데이터 학습데이터로 설정
       if i ~= j 
           for k = 1 : data_per_fold
               train_k_labels(temp*data_per_fold+k) = train_dat_l((j-1)*data_per_fold+k,:);
               train_k_imgs(temp*data_per_fold+k,:) = train_imgs_data((j-1)*data_per_fold+k,:);
               
           end
         temp = temp + 1  
       
       end
   end
   size(test_k_labels)
   size(train_k_labels)
   %학습 데이터에 대한 학습 진행
   [u_kf,v_kf,err_kf] = mlp_training_mnist(train_k_imgs, train_k_labels, alpha, lr, p, r_cnt);


   %테스트 데이터에 대해서 성능 평가 진행(전방 계산만)
   [err_tkf] = test_model_mnist(test_k_imgs, test_k_labels, alpha, lr, p, r_cnt, u_kf, v_kf);
   err_rate(i) = err_tkf; 
   total = total + err_rate(i);
end

for i = 1 : K
   err_rate(i) 
end
k_fold_err_rate = total / K
k_fold_acc = 1 - k_fold_err_rate


%%%%%%%%%%%%%%%function %%%%%%%%%%%%%%%%%%%%%%%%
%a는 sigmoid의 알파 값
function res = sigmoid(x,a)
res = 2 / (1 + exp(-a*x))-1;
end

function res = dsig(x,a)
res = a * (1+ sigmoid(x,a)) * (1 - sigmoid(x,a)) / 2;
end

function [z,z_sum] = f_sum(x,u,alpha)
x= [1 x];
[p, ~] = size(u);
z = zeros(1,p);
z_sum = zeros(1,p);
for j = 1:p
    z_sum(j) = sum(x.*u(j,:));
    z(j) = sigmoid(z_sum(j), alpha);

end
end

function [cls, dat] = feed_foward_mnist(x,u,v,alpha)
    %2-layer percepton layer이므로 전방 계산 역시 두 단계로 실시
    [z,z_sum] =  f_sum(x,u,alpha);
    [dat, dat_sum] = f_sum(z,v,alpha);
    %one hot vector에서 1값을 찾고 1값의 위치를 찾는다.
    dat_max = max(dat);
    %실제 배열은 1~10 label 0~9이기 때문에 1을 빼준다.
    cls = find (dat == dat_max) - 1;
    
end

function [u_hn, v_hn] = back_propagation_mnist(u, v, x, z, z_sum, o, o_sum, t, lr, alpha)
a = length(x);
b = length(o);
c = length(z);
delta = zeros(1,b);
du = zeros(c, a+1);
dv = zeros(b, c+1);

eta =  zeros(1,c);

x_f = [1, x];
z_f = [1, z];
%cost 함수 E를 각각 u,v로 편미분한 값이 커지는 방향으로 따라서 cost함수에서 해당 값
%을 빼주면 오류가 줄어드는 값으로 가중치를 갱신하는 Gradient descent 방식을 사용한다.
for i = 1 : b
    delta(i) = (t(i) - o(i)) * dsig(o_sum(i), alpha);
    for j = 1 : c+1
        dv(i,j) = z_f(j) * delta(i) * lr;
    end
end

for x = 1 : c
    %이 부분 이상
    eta(c) = dsig(z_sum(x), alpha) * (delta * v(:,x));
    for y =  1 : a+1
        du(x,y) = x_f(y) * eta(c) * lr;
    end
end
u_hn = u + du;
v_hn = v + dv;

end

function [u,v,err] = mlp_training_mnist(train_imgs, train_labels, alpha, lr, p, r_cnt)

%nitr은 총 데이터 개수
[nitr, trl ] = size(train_imgs);
%tel label 종류 개수
%1값을 train label크기 만큼 bitshift 시켜서 one-hot vector의 형태로 나타낸다.
one_hot_out = de2bi(bitshift(1, train_labels));
[nitr, tel ] = size(one_hot_out);  
nitr
tel%아마 레이블 개수 0~9 10개로 추정중 
err = zeros(r_cnt, 1);
u = zeros(p, trl+1);
v = zeros(tel, p+1);

cnt = 0;
%실제 레이블 값 역시 네트워크의 출력값과 비교하기 위해서는 one- hot vector의 형태로 바꿔줄 필요가 있다.


%전방 계산과 Back- propagation은 종료 조건인 반복 횟수를 채우면 해당 함수를 종료한다.
for i = 1: r_cnt
    temp = 0;
    for j = 1 : nitr
        x = train_imgs(j,:);
        t = one_hot_out(j,:);
        %전방 계산을 통해 prediction을 수행하고 비교를 수행하여 에러 여부를 판단한다.
        %one hot vector에서 1값을 찾고 1값의 위치를 찾는다.
          %실제 배열은 1~10 label 0~9이기 때문에 1을 빼준다.
        [z,z_sum] =  f_sum(x,u,alpha);
        [dat, dat_sum] = f_sum(z,v,alpha);
        dat_max = max(dat);
        cls = find (dat == dat_max) - 1;
        gt = train_labels(j);
        if(gt ~= cls)
            temp = temp+1;
        end
        %Back-Propagation을 통해 새로운 u,v값을 계산하여 업데이트 한다.
        [u,v] = back_propagation_mnist(u, v, x, z, z_sum, dat, dat_sum, t, lr, alpha);
        
    end
    cnt = cnt + 1;
    err(i) = temp / nitr;
    %scatter(cnt, err); drawnow
end
end


function [err] = test_model_mnist(test_imgs, test_labels, alpha, lr, p, r_cnt, u, v)

%nitr은 총 데이터 개수
[nitr, trl ] = size(test_imgs);
one_hot_out_test = de2bi(bitshift(1, test_labels));
[nitr, tel ] = size(one_hot_out_test);  

err = zeros(r_cnt, 1);
%u = zeros(p, trl+1);
%v = zeros(tel, p+1);

%실제 레이블 값 역시 네트워크의 출력값과 비교하기 위해서는 one- hot vector의 형태로 바꿔줄 필요가 있다.


%전방 계산과 Back- propagation은 종료 조건인 반복 횟수를 채우면 해당 함수를 종료한다.

    temp = 0;
    for j = 1 : nitr
        x = test_imgs(j,:);
        t = one_hot_out_test(j,:);
        %전방 계산을 통해 prediction을 수행하고 비교를 수행하여 에러 여부를 판단한다.(2단계)
        [z,z_sum] =  f_sum(x,u,alpha);
        [dat, dat_sum] = f_sum(z,v,alpha);
        %one hot vector에서 1값을 찾고 1값의 위치를 찾는다.
        dat_max = max(dat);
          %실제 배열은 1~10 label 0~9이기 때문에 1을 빼준다.
        cls = find (dat == dat_max) - 1;
        gt =  test_labels(j);
        if(gt ~= cls)
            temp = temp+1;
        end
        %Back-Propagation을 통해 새로운 u,v값을 계산하여 업데이트 한다.
        %[u,v] = back_propagation_mnist(u, v, x, z, z_sum, dat, dat_sum, t, lr, alpha);
        
    end

    err = temp / nitr;
    %scatter(cnt, err); drawnow

end

