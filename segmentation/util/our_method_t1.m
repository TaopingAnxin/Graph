function  seg  = our_method_t1( u_ori,W,edges, a,b,is_square,threshold_res)
%%%%%%%%%%%%%%%%%%%%%  smooth image  %%%%%%%%%%%%%%
if nargin <7
        threshold_res = 1e-6;
end
if nargin <5
    b = 0.1;  %  0.1
end
if nargin < 4
    a = 1; 
end
if nargin < 6
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
        k = TotalCuNew(u);
        afa = a + b*(abs(k));  



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




