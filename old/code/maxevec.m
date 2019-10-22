%Given square matrix a, its max eigenvalue mu=mu(a) and an
%eigenvector evec is determined. mu is calculated by Karp’s
%formula. The eigenvector evec is calculated by the
%Floyd-Warshall procedure, as the first column of b^+,
%b=a/mu(a) which has diagonal entry 1.
function[mu,evec]=maxevec(a)
	[m,n]=size(a);
	z=[];
	for i=1:n
		z(1,i)=0;
	end;
	z(1,1)=1;
	% z_1 = [1,0, ... <n>, 0]
	
	for i=2:n+1
		w=z(i-1,:);
		w=max(diag(w)*a’);
		z=[z;w];
	end

	z1=z*diag(w.^-1);
	z1 =z1(1:n,:);
	for i=1:n
		t= z1(i,:);
		z1(i,:) = t.^(1/(n+1-i));
	end
	
	mu=min(max(z1)); mu =mu^-1;
	b=mu^{-1}*a;

	for i=1:n
		w=b(:,i);
		u= b(i,:);
			c=w*u;
		for j=1:n
			for k=1:n
				b(j,k)=max(c(j,k),b(j,k));
			end;
		end;
	end;
	tol=10^-10;
	for i=1:n
		if abs(b(i,i)-1) < tol
			evec=b(:,i); 
			return;
	end;
end;
