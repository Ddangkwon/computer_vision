close all;
cnt= 0;
train_pca = [];

test_folder_num = 5; 
test_data_num = 2;
for i = 1 : test_folder_num 
    for j = 9 : 10
        fd= i;
        fp = j;
        [testImage,images,H,W,M,m,U,cnt_,pca_vec]=train_procedure(fd,fp);
        cnt = cnt + cnt_;
    end
end
%accuracy ���� part ��ü test set�� ���ؼ� ���� ���� �� ������ ��(������ �̹����� test�̹����� ���� ������
%���� �� ���� ������� ����Ű�Ƿ� ���������� ���� ����

acc = (cnt / (test_folder_num *test_data_num))
train_pca = [train_pca pca_vec];

min = 9999;
idx = 0;




function [testImage,images,H,W,M,m,U,cnt_,pca_vec]=train_procedure(fd,fp)
    %�н������� ������ ����
    %�� training set �̹��� 320�� test set�� 80���� Ȱ���� ����
    M = 400;
    cnt_= 0;
    %HW*M matrix 
    vec=zeros(10304,M);
    im = imread('C:/Users/dram109/Desktop/codeAndfile_2015124010/s1/2.pgm');
    H=size(im,1)%Height 
    W=size(im,2) %Width
    images=zeros(H,W,M);
    
    testImage = strcat('C:/Users/dram109/Desktop/codeAndfile_2015124010/s', num2str(fd), '/', num2str(fp),'.pgm');
    %testImage = ('C:/Users/dram109/Desktop/Faces/test.pgm');
   %testImage = imread('C:/Users/dram109/Desktop/Faces/s1/2.pgm');
    vec=zeros(H*W,M);
    %������ �ǹ�(������ �̹��� 10���� ����)
    train_num = 40;
    X = [];
    for i=1:train_num
        %���� �˻� s1,s2....
        cd(strcat('s',num2str(i)));
        for j=1:10
            %���� �� �̹��� �˻�..
            a=imread(strcat(num2str(j),'.pgm'));
            images(:,:,(i-1)*10+j) = a;
            %(H,W,1) =>
            vec(:,(i-1)*10+j)=reshape(a,size(a,1)*size(a,2),1);
           
        end
        cd ..
    end

    % mean face(��� ���� fav)
    m=sum(vec,2)/M;
    % face space
    A=vec-repmat(m,1,M);
    
    L=A'*A;
    
    [V,lambda]=eig(L);
    size(V)
    size(lambda)
    % eigenvector of the covariance matrix of A. These are the eigenfaces
    U=A*V;
    eigenface=[];
    for k=1:M
        c  = U(:,k);
        eigenface{k} = reshape(c,H,W);
    end
    size(eigenface{1})
    x = zeros(300,1);
    for i = 1 : size(lambda,1)
        
         x(i,1) =lambda(i,i);
    end

    size(x);
    [xc,xci]=sort(x,'descend');%�������� ����
    
    xc(1) 
    %�Ʒ��� ���� bubble sort�� ��ü ����(�Ȱ��� �������� ���� ����)
    %{
    for i = 1 : size(x)
        for j = 1 : size(x) - i
            temp = 0;
            if (x(j) < x(j+1)) 
            
                temp= x(j);
                x = x(j+1);
                x(j+1) = temp;
            end
          
        end
         
    end
    
    %}
    size(xc)
    size(xci)
  
    L_eig_vec = [];
    pca_vec = [];
    %����� ���� ���Ͱ��� ��� ���� ����
    %count �� xc�� ������� ������� �����ش�.
    %L_eig_vec �迭�� lambda  eigen vector���� ��� �־��ش�.
    count = 0;
    for i = 1 : size(V,2) 
        if( lambda(i,i) > xc(100) )
            L_eig_vec = [L_eig_vec V(:,i)];
           
            count = count + 1;
        end
   end
   
    answer = size( L_eig_vec)
    size(A);
    %%% finally the eigenfaces %%%
    eigenfaces = A * L_eig_vec;

    projectimg = [ ];  % projected image vector matrix
    for i = 1 : size(eigenfaces,2)
        temp = eigenfaces' * A(:,i);
        projectimg = [projectimg temp];
    end
    size(projectimg)
    %%%%% extractiing PCA features of the test image %%%%%
    test_image = imread(testImage);
    %figure
    %imshow(testImage);
    test_image = test_image(:,:,1);
    [HW WI] = size(test_image);
    %test_img���� M*N.1 ��� ��ȯ
    temp = reshape(test_image',HW*WI,1);
    %mean vector�� ���ش�.
    temp = double(temp)-m; 
    projtestimg = eigenfaces'*temp;
 
    euclide_dist = [ ];
    for i=1 : size(eigenfaces,2)
        temp = (norm(projtestimg-projectimg(:,i)))^2;
        euclide_dist = [euclide_dist temp];
    end
    
    [euclide_dist_min recognized_index] = min(euclide_dist)
    figure
    subplot(1,2,1)
    imshow(testImage)
    title('Test set face')
    subplot(1,2,2)
    imshow(uint8(images(:,:,recognized_index)))
    recognized_index
    folder =  floor(recognized_index / 10) + 1
    fd
    %������ �̹����� �׽�Ʈ �̹����� ���� �̹����� ���� �� +1
    if folder == fd
        cnt_ = cnt_ + 1;
    end
    cnt_
    img_num = mod(recognized_index ,10);
    title(['Find in training datset s',num2str(folder), ' ', num2str(img_num), '.pgm'])
 
end

function [testImage,images,H,W,M,m,U,cnt_,pca_vec]=test_procedure(fd,fp)
    %�н������� ������ ����
    %�� training set �̹��� 320�� test set�� 80���� Ȱ���� ����
    M = 400;
    cnt_= 0;
    %HW*M matrix 
    vec=zeros(10304,M);
    im = imread('C:/Users/dram109/Desktop/Faces/s1/2.pgm');
    H=size(im,1)%Height 
    W=size(im,2) %Width
    images=zeros(H,W,M);
    
    testImage = strcat('C:/Users/dram109/Desktop/Faces/s', num2str(fd), '/', num2str(fp),'.pgm');
   %testImage = ('C:/Users/dram109/Desktop/Faces/test.pgm');
    vec=zeros(H*W,M);
    %������ �ǹ�(������ �̹��� 10���� ����)
    train_num = 40;
    for i=1:train_num
        %���� �˻� s1,s2....
        cd(strcat('s',num2str(i)));
        for j=9:10
            %���� �� �̹��� �˻�..
            answer = (i-1)*10+j
            a=imread(strcat(num2str(j),'.pgm'));
            images(:,:,(i-1)*10+j) = a;
            %(H,W,1) =>
            vec(:,(i-1)*10+j)=reshape(a,size(a,1)*size(a,2),1);
           
        end
        cd ..
    end
    
   
    % mean face
    m=sum(vec,2)/M;
    % face space
    A=vec-repmat(m,1,M);
    
    L=A'*A;
    
    [V,lambda]=eig(L);
    size(V)
    size(lambda)
    % eigenvector of the covariance matrix of A. These are the eigenfaces
    U=A*V;
    eigenface=[];
    for k=1:M
        c  = U(:,k);
        eigenface{k} = reshape(c,H,W);
    end
    size(eigenface{1})
    x = zeros(300,1);
    for i = 1 : size(lambda,1)
        
         x(i,1) =lambda(i,i);
    end
 
    size(x);
    [xc,xci]=sort(x,'descend');%�������� ����
    
    xc(1) 
    %�Ʒ��� ���� bubble sort�� ��ü ����(�Ȱ��� �������� ���� ����)
    %{
    for i = 1 : size(x)
        for j = 1 : size(x) - i
            temp = 0;
            if (x(j) < x(j+1)) 
            
                temp= x(j);
                x = x(j+1);
                x(j+1) = temp;
            end
          
        end
         
    end
    
    %}
    size(xc)
    size(xci)
  
    L_eig_vec = [];
    pca_vec = [];
    %����� ���� ���Ͱ��� ��� ���� ����
    %count �� xc�� ������� ������� �����ش�.
    %L_eig_vec �迭�� lambda  eigen vector���� ��� �־��ش�.
    count = 0;
    for i = 1 : size(V,2) 
        if( lambda(i,i) > xc(50) )
            L_eig_vec = [L_eig_vec V(:,i)];
           
            count = count + 1;
        end
   end
    count
    i = 1;
    while(count)
       i = i + 1;
        pca_vec = [pca_vec xc(i)];
        count = count - 1;
    end
    size( pca_vec)
    size(A);
    %%% finally the eigenfaces %%%
    eigenfaces = A * L_eig_vec;

    projectimg = [ ];  % projected image vector matrix
    for i = 1 : size(eigenfaces,2)
        temp = eigenfaces' * A(:,i);
        projectimg = [projectimg temp];
    end
    %%%%% extractiing PCA features of the test image %%%%%
    test_image = imread(testImage);
    %figure
    %imshow(testImage);
    test_image = test_image(:,:,1);
    [r c] = size(test_image);
    temp = reshape(test_image',r*c,1); % creating (MxN)x1 image vector from the 2D image
    temp = double(temp)-m; % mean subtracted vector
    projtestimg = eigenfaces'*temp; % projection of test image onto the facespace
    %%%%% calculating & comparing the euclidian distance of all projected trained images from the projected test image %%%%%
    euclide_dist = [ ];
    for i=1 : size(eigenfaces,2)
        temp = (norm(projtestimg-projectimg(:,i)))^2;
        euclide_dist = [euclide_dist temp];
    end
    [euclide_dist_min recognized_index] = min(euclide_dist);
    figure
   subplot(1,2,1)
    imshow(testImage)
    title('Test set face')
    recognized_index
    
      folder = floor(recognized_index/ 10) 
    

    fd
    %������ �̹����� �׽�Ʈ �̹����� ���� �̹����� ���� �� +1
    if folder == fd
        cnt_ = cnt_ + 1;
    end
    cnt_
    
    img_num = mod(recognized_index ,10);
   
    subplot(1,2,2)
    imshow(uint8(images(:,:,recognized_index)))
    title(['Find in training datset s',num2str(folder), ' ', num2str(img_num), '.pgm'])
 
end
