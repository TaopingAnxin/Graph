function  seg  = our_method_t( u_ori,W,edges,is_square,afa)

%%%%%%%%%%%%%%%%%%%%%  smooth image  %%%%%%%%%%%%%%
% lambda = 2; %%3.0
% p1= 1.0; %  1.0  0.5   1                                 
% p2= 6.0; % 6.0 4
% p3 = 0.2; % 6
% % delta = 1e-6; %
if nargin < 5
    afa = 3; 
end

if nargin < 4
    is_square = 1;
end

if is_square == 1
    lambda = 2; %%3.0
    p1= 1.0; %  1.0  0.5   1                                 
    p2= 6.0; % 6.0 4
    p3 = 0.2; % 6
    % delta = 1e-6; %
else
    lambda = 0.5; %%3.0
    p1= 1.0; %  1.0  0.5   1                                 
    p2= 6.0; % 6.0 4
    p3 = 0.01;
end
                         
num_iter = 500;
threshold_res = 1e-6;

[Ny,Nx] = size(u_ori); 

[l1, l2] = size(u_ori);
% Area = l1*l2;

u0 = u_ori;
u = u_ori*0;
v = u;

lambda11 = u0*0; lambda12 = u0*0; 
lambda2 = u0*0;

 [x1,x2] = gradient(u_ori);
% x1 = u0*0;
% x2 = u0*0; 

DtD = abs(psf2otf([1,-1],[l1, l2])).^2 + abs(psf2otf([1;-1],[l1, l2])).^2;
A = (p1)*DtD + p2;

ave_c = zeros(2,num_iter);
% residual = zeros(2,num_iter);
% rel_error_L = zeros(2,num_iter);
% rel_error_u = zeros(1,num_iter);
% energy = zeros(1,num_iter);
% rel_error_L(1:2,1) = 1;
% rel_error_u(1) = 1;

% real_iter = num_iter;
% record_flag = 0;

% lambda1_L1 = 1;
% lambda2_L1 = 1;


% afa = 3*ones(Ny,Nx);
% afa_delta = zeros(Ny,Nx,2);
x_delta = zeros(Ny,Nx,2);

t = cputime;
for iter=1:num_iter
    %%%%%%%%%%%%%%%%%%%  For c1 and c2
    [c1,c2] = Cal_Averages(u_ori,v);
    ave_c(1,iter) = c1;
%     c1 = c1 + wailine;
    ave_c(2,iter) = c2;  
%     c2 = c2 + line;
    
    %%%%%%%%%%%%%%%%%%%%%  For u    
    u_old = u;
    
    g = - dxb(p1*x1 + lambda11) - dyb(p1*x2 + lambda12) + p2*v +lambda2;
    g = fftn(g);
    u = real(ifftn(g./A));
    
    %%%%%%%%%%%%%%%%%%%%%  For v
    temp_b = lambda*((u_ori-c1).^2-(u_ori-c2).^2) +lambda2;
    temp_b = reshape(temp_b,size(temp_b,1)*size(temp_b,2),1);
    dd = sum(W);
    v = gauss_seidel(W,edges,dd,p2,p3,temp_b,u,v);
    
    
    %%%%%%%%%%%%%%%%%%%%%  Update afa
%     k = TotalCuNew(u);
%     afa = a + b*(abs(k));  
%  our method 
%      if iter< 10
%           afa = afa - 0.01*sqrt(x1.^2 + x2.^2);% 0.01
%         else 
%           afa = afa - 0.0001*sqrt(x1.^2 + x2.^2);% 0.0001
%      end
    
%    if iter == 1
%        afa = afa - delta*sqrt(x1.^2 + x2.^2);
%        afa_delta(:,:,2) = afa;
%    elseif iter == 2
%        afa = afa - delta*sqrt(x1.^2 + x2.^2);
%        afa_delta(:,:,1) = afa;
%    else
%        sk = afa_delta(:,:,1) - afa_delta(:,:,2);
%        gk = x_delta(:,:,1) - x_delta(:,:,2);
%        temp1 = sk.*gk;
%        delta = sum(temp1(:));
%        temp2 = gk.*gk;
%        delta2 = sum(temp2(:))+(1e-10);
%        delta = delta/delta2;
%        afa = afa - delta*sqrt(x1.^2 + x2.^2);
%        afa_delta(:,:,2) = afa_delta(:,:,1);
%        afa_delta(:,:,1) = afa;
%     end
       


    %%%%%%%%%%%%%%%%  For x
    
    xx = dxf(u) - lambda11/p1;
    xy = dyf(u) - lambda12/p1;
    x = sqrt(xx.^2 + xy.^2);
    x(x==0) = 1;
    x = max(x - afa/p1,0)./x;
    x1 = xx.*x;
    x2 = xy.*x;

    if iter == 1
       x_delta(:,:,2) = sqrt(x1.^2 + x2.^2);
   elseif iter == 2
       x_delta(:,:,1) = sqrt(x1.^2 + x2.^2);
    else
        x_delta(:,:,2) = x_delta(:,:,1);
        x_delta(:,:,1) = sqrt(x1.^2 + x2.^2);
    end
    
    %%%%%%%%%%%%%%%%%%%%%  For Lambda    
%     lambda11_old = lambda11;
%     lambda12_old = lambda12;
%     lambda2_old = lambda2; 
    
    lambda11 = lambda11 + p1*(x1 - dxf(u));
    lambda12 = lambda12 + p1*(x2 - dyf(u));
    lambda2  = lambda2  + p2*(v-u);
    
%     R11 = x1 - dxf(u);
%     R12 = x2 - dyf(u);
%     R2  = v - u;
    
    error1 =  sum(sum( abs(u-u_old) ))/sum(sum(abs(u_old))); 
    
    if error1<threshold_res
        error1
        iter
        break
    end
    
%     %%%%%%%%%%%%%%%%%%%%%  For residual
%     
%     residual(1,iter) = sum( abs(R11(:))+abs(R12(:)) )/Area;
%     residual(2,iter) = sum( abs(R2(:)) )/Area;    
%     
%     %%%%%%%%%%%%%%%%%%%%%  For relative error
% 
%     rel_error_u(iter) = sum(sum( abs(u-u_old) ))/Area;
%     
%     rel_error_L(1,iter) = p1*residual(1,iter)/lambda1_L1;
%     rel_error_L(2,iter) = p2*residual(2,iter)/lambda2_L1;
% 
%     max_rel_error = max( residual(:,iter) );
%     if( max_rel_error < threshold_res )
%         if( record_flag==0 )
%             real_iter = iter;
%             record_flag = 1;
%         end
%     end
%     
%     % L1 norm of Lagrange multipliers.
%     if( iter>2 )
%         lambda1_L1 = sum( abs(lambda11(:))+abs(lambda12(:)) )/Area;
%         lambda2_L1 = sum( abs(lambda2(:)) )/Area;
% %         u_L1 = sum( abs(u(:)) );
%     end
% 
%      energy(iter) = Copy_of_TCEuler_energy(v,u_ori,x1,x2,afa,lambda,c1,c2);
end
t = cputime - t;
seg = im2bw(1-u);
end
function v = gauss_seidel(W,edges,dd,p2,p3,temp_b,u,v)
  v = reshape(v,size(v,1)*size(v,2),1);
  iters = 5;
  [m,n]=size(u);
  y = reshape(u,m*n,1);  
  dd = dd';
  md = 1./sqrt(dd);
  md2 = 1./dd;
  clear dd;
  Wp = sparse(edges(:,1),edges(:,2), md(edges(:,1)));
  Wq = sparse(edges(:,1),edges(:,2), md(edges(:,2)));
  Wd = sparse(edges(:,1),edges(:,2), md2(edges(:,1))); 

  c=zeros([m*n,1]);
  WC = W.*Wq;
  WC = WC.*Wp;
  WD = W.*Wd;
  d = sum(WD,2)*p3+p2;
  
  for i =1:iters
        c = WC*v*p3;
        c = c + p2*y-temp_b;
        v = c./d;
        v=max(min(v,1),0);
  end
  v = reshape(v,m,n);
end




