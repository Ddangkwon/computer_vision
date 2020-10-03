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
%N�� TRAIN DATA����, M�� �׽�Ʈ ������ ����
N = 600;
M = 1000;
train_labels = zeros(N, 1);
test_labels = zeros(M, 1);
train_imgs = zeros(N, 784);
test_imgs = zeros(M, 784);
%�н� �����Ϳ� �Ʒ� ������ ũ�� ����
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


%���� ��� �׽�Ʈ
%[cls, dat] = feed_foward_mnist((train_imgs(1,:)),u,v,alpha);

alpha = 0.1;
lr =  0.05;
%p�� ��� �������� ��� ������ Ī�Ѵ�.(inp
p =80;
r_cnt = 200;
%�н� �����Ϳ� ���� �н� ����
[u,v,err] = mlp_training_mnist(train_imgs, train_labels, alpha, lr, p, r_cnt);


%�׽�Ʈ �����Ϳ� ���ؼ� ���� �� ����(���� ��길)
[err_t] = test_model_mnist(test_imgs, test_labels, alpha, lr, p, r_cnt, u, v);
cnt = 1 : 1 : r_cnt;
%�н� ��� Plot
figure 
plot(cnt, err, '-or');
title('training result')
xlabel('epoch')
ylabel('error rate')
final_error_rate = err(r_cnt)
%�׽�Ʈ ���� �� ���


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
        %�׽�Ʈ �����͸� ���� �̴´�.
        test_k_labels(k) = train_dat_l((i-1)*data_per_fold +k,:);
        test_k_imgs(k,:) = train_imgs_data((i-1)*data_per_fold +k,:);
    end
    temp = 0;
   for j = 1 : K
       %�׽�Ʈ �����ͷ� ���� �����͸� ������ ��� ������ �н������ͷ� ����
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
   %�н� �����Ϳ� ���� �н� ����
   [u_kf,v_kf,err_kf] = mlp_training_mnist(train_k_imgs, train_k_labels, alpha, lr, p, r_cnt);


   %�׽�Ʈ �����Ϳ� ���ؼ� ���� �� ����(���� ��길)
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
%a�� sigmoid�� ���� ��
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
    %2-layer percepton layer�̹Ƿ� ���� ��� ���� �� �ܰ�� �ǽ�
    [z,z_sum] =  f_sum(x,u,alpha);
    [dat, dat_sum] = f_sum(z,v,alpha);
    %one hot vector���� 1���� ã�� 1���� ��ġ�� ã�´�.
    dat_max = max(dat);
    %���� �迭�� 1~10 label 0~9�̱� ������ 1�� ���ش�.
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
%cost �Լ� E�� ���� u,v�� ��̺��� ���� Ŀ���� �������� ���� cost�Լ����� �ش� ��
%�� ���ָ� ������ �پ��� ������ ����ġ�� �����ϴ� Gradient descent ����� ����Ѵ�.
for i = 1 : b
    delta(i) = (t(i) - o(i)) * dsig(o_sum(i), alpha);
    for j = 1 : c+1
        dv(i,j) = z_f(j) * delta(i) * lr;
    end
end

for x = 1 : c
    %�� �κ� �̻�
    eta(c) = dsig(z_sum(x), alpha) * (delta * v(:,x));
    for y =  1 : a+1
        du(x,y) = x_f(y) * eta(c) * lr;
    end
end
u_hn = u + du;
v_hn = v + dv;

end

function [u,v,err] = mlp_training_mnist(train_imgs, train_labels, alpha, lr, p, r_cnt)

%nitr�� �� ������ ����
[nitr, trl ] = size(train_imgs);
%tel label ���� ����
%1���� train labelũ�� ��ŭ bitshift ���Ѽ� one-hot vector�� ���·� ��Ÿ����.
one_hot_out = de2bi(bitshift(1, train_labels));
[nitr, tel ] = size(one_hot_out);  
nitr
tel%�Ƹ� ���̺� ���� 0~9 10���� ������ 
err = zeros(r_cnt, 1);
u = zeros(p, trl+1);
v = zeros(tel, p+1);

cnt = 0;
%���� ���̺� �� ���� ��Ʈ��ũ�� ��°��� ���ϱ� ���ؼ��� one- hot vector�� ���·� �ٲ��� �ʿ䰡 �ִ�.


%���� ���� Back- propagation�� ���� ������ �ݺ� Ƚ���� ä��� �ش� �Լ��� �����Ѵ�.
for i = 1: r_cnt
    temp = 0;
    for j = 1 : nitr
        x = train_imgs(j,:);
        t = one_hot_out(j,:);
        %���� ����� ���� prediction�� �����ϰ� �񱳸� �����Ͽ� ���� ���θ� �Ǵ��Ѵ�.
        %one hot vector���� 1���� ã�� 1���� ��ġ�� ã�´�.
          %���� �迭�� 1~10 label 0~9�̱� ������ 1�� ���ش�.
        [z,z_sum] =  f_sum(x,u,alpha);
        [dat, dat_sum] = f_sum(z,v,alpha);
        dat_max = max(dat);
        cls = find (dat == dat_max) - 1;
        gt = train_labels(j);
        if(gt ~= cls)
            temp = temp+1;
        end
        %Back-Propagation�� ���� ���ο� u,v���� ����Ͽ� ������Ʈ �Ѵ�.
        [u,v] = back_propagation_mnist(u, v, x, z, z_sum, dat, dat_sum, t, lr, alpha);
        
    end
    cnt = cnt + 1;
    err(i) = temp / nitr;
    %scatter(cnt, err); drawnow
end
end


function [err] = test_model_mnist(test_imgs, test_labels, alpha, lr, p, r_cnt, u, v)

%nitr�� �� ������ ����
[nitr, trl ] = size(test_imgs);
one_hot_out_test = de2bi(bitshift(1, test_labels));
[nitr, tel ] = size(one_hot_out_test);  

err = zeros(r_cnt, 1);
%u = zeros(p, trl+1);
%v = zeros(tel, p+1);

%���� ���̺� �� ���� ��Ʈ��ũ�� ��°��� ���ϱ� ���ؼ��� one- hot vector�� ���·� �ٲ��� �ʿ䰡 �ִ�.


%���� ���� Back- propagation�� ���� ������ �ݺ� Ƚ���� ä��� �ش� �Լ��� �����Ѵ�.

    temp = 0;
    for j = 1 : nitr
        x = test_imgs(j,:);
        t = one_hot_out_test(j,:);
        %���� ����� ���� prediction�� �����ϰ� �񱳸� �����Ͽ� ���� ���θ� �Ǵ��Ѵ�.(2�ܰ�)
        [z,z_sum] =  f_sum(x,u,alpha);
        [dat, dat_sum] = f_sum(z,v,alpha);
        %one hot vector���� 1���� ã�� 1���� ��ġ�� ã�´�.
        dat_max = max(dat);
          %���� �迭�� 1~10 label 0~9�̱� ������ 1�� ���ش�.
        cls = find (dat == dat_max) - 1;
        gt =  test_labels(j);
        if(gt ~= cls)
            temp = temp+1;
        end
        %Back-Propagation�� ���� ���ο� u,v���� ����Ͽ� ������Ʈ �Ѵ�.
        %[u,v] = back_propagation_mnist(u, v, x, z, z_sum, dat, dat_sum, t, lr, alpha);
        
    end

    err = temp / nitr;
    %scatter(cnt, err); drawnow

end

