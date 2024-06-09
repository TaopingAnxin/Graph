close all
clear all

addpath('img'); 
addpath('util');



%%%%%%%%%%%%%%%%%%%%%  lena pepper   %%%%%%%%%%%%%%
lambda = 1; 
p1= 20；                                      
a = 1;                                                   
b = 28;       
p3 = 0.1;
para = 21;                                    
theta=10; % control the edge weight

iter = 300;
threshold_res = 5e-5;

uclean = double(imread('im121.jpg'));                  
       

sigma = 30;
u0 = uclean+sigma*randn(size(uclean));
[Ny,Nx,Nc] = size(u0); 
%%%%%%%%%%
lab_vals = colorspace('Lab<-', u0); 
lab_vals = reshape(lab_vals,size(lab_vals,1)*size(lab_vals,2),size(lab_vals,3));


i1 = 1:Ny;
edi1 = i1';
edi1 = repmat(edi1,Ny,1);
i2 = 1:Ny;
edi2 = repmat(i2,Ny,1);
edi2 = edi2(:);

j1 = 1:Nx;
edj1 = j1';
edj1 = repmat(edj1,Nx,1);
j2 = 1:Nx;
edj2 = repmat(j2,Nx,1);
edj2 = edj2(:);

ii =[edi1 edi2];
jj =[edj1 edj2];
num3 = (ii(:,1)==ii(:,2));
ii(num3,:)=[];
num3 = (jj(:,1)==jj(:,2));
jj(num3,:)=[];

num1 = find(abs(ii(:,1)-ii(:,2))<5);
num2 = find(abs(jj(:,1)-jj(:,2))<5);
edij1 = repmat(ii(num1,:),size(num2,1),1);
temp = jj(num2,:);
temp=temp';
temp1 = repmat(temp(1,:),size(num1,1),1);
temp1 = temp1(:);
temp2 = repmat(temp(2,:),size(num1,1),1);
temp2 = temp2(:);
edij2 = [temp1 temp2];
e1 = (edij2(:,1)-1)*Ny+edij1(:,1);
e2 = (edij2(:,2)-1)*Ny+edij1(:,2);
edges = [e1 e2];

clear edi1;
clear edi2;
clear edj1;
clear edj2;
clear edij1;
clear edij2;
 
weights = makeweights(edges,lab_vals,theta);
W=sparse(edges(:,1),edges(:,2), weights);

%% TC + graph
a = 1;
b = 28;
t1=clock;
u(:,:,1) = spdenosing_t1(W,edges,u0(:,:,1),iter,lambda,p1,p3,a,b, threshold_res);
u(:,:,2) = spdenosing_t1(W,edges,u0(:,:,2),iter,lambda,p1,p3,a,b, threshold_res);
u(:,:,3) = spdenosing_t1(W,edges,u0(:,:,3),iter,lambda,p1,p3,a,b, threshold_res);
t2=clock;

t=etime(t2,t1);
psnr_u = psnr(uint8(u),uint8(uclean))
ssim_u = ssim(uint8(u),uint8(uclean))
figure;
imshow(uint8(u));
title('The result');
%% TV + graph
t3=clock;
u1(:,:,1) = spdenosing_t(W,edges,u0(:,:,1),iter,lambda,p1,p3,para, threshold_res);
u1(:,:,2) = spdenosing_t(W,edges,u0(:,:,2),iter,lambda,p1,p3,para, threshold_res);
u1(:,:,3) = spdenosing_t(W,edges,u0(:,:,3),iter,lambda,p1,p3,para, threshold_res);
t4=clock;

t5=etime(t4,t3);

psnr_u1 = psnr(uint8(u1),uint8(uclean))
ssim_u1 = ssim(uint8(u1),uint8(uclean))
figure;
imshow(uint8(u1));
title('The result');
