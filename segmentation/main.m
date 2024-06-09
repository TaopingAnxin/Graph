close all
clear all

addpath('img'); 
% addpath('test'); 
addpath('util');

% Im0 = imread('C:/Users/Administrator/Desktop/seg/smooth-seg/inpainting/img/fig8.png');
% figure;
% imshow(Im0,[],'border','tight');
is_square = 1;
if is_square == 1
    Im0 = imread('130.bmp');
else
    Im0 = imread('120.jpg');
end

Im0 = imresize(Im0,2);
exact = im2bw(Im0);
[Ny,Nx,Nc] = size(Im0); 
if Nc>1; Im0=rgb2gray(Im0); end; % Convert color image to gray-scale image
%imwrite(Im0,'3096-1.png');
Im0 = double(Im0);

u_ori = Im0/255;
sigma = 0.8;

% u_ori = Im0;
% sigma = 40;
u_ori = u_ori+sigma*randn(size(u_ori));

% exact = imread('130.bmp');
% exact = imresize(exact,2);
%% %%%%%%%%%%%%%% W_{ij} %%%%%%%%%%%%%%%%%%
u_ori_rgb = zeros(Ny,Nx,Nc);
u_ori_rgb(:,:,1) = u_ori;
u_ori_rgb(:,:,2) = u_ori;
u_ori_rgb(:,:,3) = u_ori;
lab_vals = colorspace('Lab<-', u_ori_rgb); 
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
num3 = ii(:,1)==ii(:,2);
ii(num3,:)=[];
num3 = jj(:,1)==jj(:,2);
jj(num3,:)=[];

num1 = find(abs(ii(:,1)-ii(:,2))<2);
num2 = find(abs(jj(:,1)-jj(:,2))<2);
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

theta=10; % control the edge weight 
weights = makeweights(edges,lab_vals,theta);
W=sparse(edges(:,1),edges(:,2), weights);

%% ##############our method #########
a = 1;
b = 5;
t10=clock;
seg0 = our_method_t1( u_ori,W,edges,a,b,is_square); % TC+graph
t20=clock;

t0=etime(t20,t10);
%seg_show = seg;

t1=clock;
seg = our_method_t( u_ori,W,edges,is_square); % TV +graph
t2=clock;

t=etime(t2,t1);
% t11=clock;
% seg11 = our_method( u_ori,W,edges); % pre
% t21=clock;
% 
% t_our=etime(t21,t11);
% % seg  = our_method( u_ori,W,edges )
%###############################

if is_square == 0
    fprintf(' ***********tc + graph *********** ')
    seg_show0 = 1-seg0;
    [pre0 ,rec0, JS0] = pre_rec(seg_show0,exact)
    % [pre0 ,rec0, JS0] = pre_rec(seg0,exact)

    figure,imshow(seg_show0,[],'border','tight');
    fprintf(' The iteration time is: %4.2fs', t0);


    fprintf(' *********** TV + graph *********** ')
    seg_show = 1-seg;
    [pre ,rec, JS] = pre_rec(seg_show,exact)
    figure,imshow(seg_show,[],'border','tight');
    fprintf(' The iteration time is: %4.2fs', t);

    else
    fprintf(' ***********tc + graph *********** ')
    [pre0 ,rec0, JS0] = pre_rec(seg0,exact)
    % [pre0 ,rec0, JS0] = pre_rec(seg0,exact)
    seg_show0 = 1-seg0;
    figure,imshow(seg_show0,[],'border','tight');
    fprintf(' The iteration time is: %4.2fs', t0);


    fprintf(' *********** TV + graph *********** ')
    [pre ,rec, JS] = pre_rec(seg,exact)
    seg_show = 1-seg;
    figure,imshow(seg_show,[],'border','tight');
    fprintf(' The iteration time is: %4.2fs', t);

 end

%%
function weights=makeweights(edges,vals,valScale)
valDistances=sqrt(sum((vals(edges(:,1),:)-vals(edges(:,2),:)).^2,2));
valDistances=normalize(valDistances); %Normalize to [0,1]
weights=exp(-valScale*valDistances);
end
