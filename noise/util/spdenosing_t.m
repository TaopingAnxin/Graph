function [u,relative_error] = spdenosing_t(W,edges,u0,iter,lambda,p1,p3,afa, threshold_res) 

%     [l1, l2, l3] = size(u0);

    lambda11 = u0*0; lambda12 = u0*0; 
    [x1,x2] = gradient(u0);
    %x1 = u0*0; x2 = u0*0; 
    u = u0;
    
    relative_error = zeros(1,iter);

%     real_iter = 1;
%     record_flag = 0;
%     afa =para * ones(l1,l2,l3);

    for i=1:iter
        u_old = u;
        g = lambda*u0 - dxb(p1*x1 + lambda11) - dyb(p1*x2 + lambda12);

        u = gauss_seidel(W,edges,lambda,p1,p3,g,u);

        %%%%%%%%%%%%%%%%  For x

        xx = dxf(u) - lambda11/p1;
        xy = dyf(u) - lambda12/p1;
        x = sqrt(xx.^2 + xy.^2);
        x(x==0) = 1;
        x = max(x - afa/p1,0)./x;
        x1 = xx.*x;
        x2 = xy.*x;

         %%%%%%%%%%%%%%%%  update lambda
        lambda11 = lambda11 + p1*(x1 - dxf(u));
        lambda12 = lambda12 + p1*(x2 - dyf(u));

    %     energy(i) = TCEuler_energy(u,u0,x1,x2,afa,lambda);
    %     
    %     residual(i) = sum( abs(R11(:))+abs(R12(:)) )/Area;
    %     error(i) = sum(sum(abs(lambda11-lambda11_old)+abs(lambda12-lambda12_old)))/Area;
        relative_error(i) =  sum(sum( abs(u-u_old) ))/sum(sum(u_old)); 
        
          if( relative_error(i) < threshold_res )
%             if( record_flag==0 )
%                 real_iter = i;
%                 record_flag = 1;
%             end
            break;
          end   
    end
end

function u = gauss_seidel(W,edges,lambda,p1,p3,g,u)

%%%%%%%%%%%%%%
  [m,n,p]=size(u);
  y = reshape(u,m*n,p);  
  
  temp = u([end 1:end-1],:,:) + u([2:end 1],:,:) + u(:,[end 1:end-1],:) + u(:,[2:end 1],:);
  temp = p1*temp + g;
  temp = reshape(temp,m*n,p);
  dd = sum(W); 
  dd = dd';
  md = 1./sqrt(dd);
  md2 = 1./dd;
  clear dd;
  Wp = sparse(edges(:,1),edges(:,2), md(edges(:,1)));
  Wq = sparse(edges(:,1),edges(:,2), md(edges(:,2)));
  Wd = sparse(edges(:,1),edges(:,2), md2(edges(:,1))); 

  WC = W.*Wq;
  WC = WC.*Wp;
  WD = W.*Wd;
  d = sum(WD,2)*p3;
  d = d + lambda + p1*4;
 
  c = WC*y*p3;
  c = c + temp;
  c = c./d;
  u = reshape(c,m,n,p);
end