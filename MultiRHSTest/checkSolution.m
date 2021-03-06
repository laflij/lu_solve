close all
clear all
clc

% Read the mavtrix, RHS array, and solution arrays.
A = mmread('./build/A.mtx');
RHS = mmread('./build/RHS.mtx');
Y = mmread('./build/Y.mtx');
X = mmread('./build/X.mtx');

% Figure out problem dimensions.
N = size(A,1);
R = length(RHS)/N;
[ia,ja] = find(A);
K = max(ia-ja);

% Reshape RHS, Y, and X
RHS = reshape(RHS,N,R);
Y = reshape(Y,N,R);
X = reshape(X,N,R);

% Extract L and U factors from A.
L = spdiags([ones(N,1) zeros(N,K)], (0:K), A);
U = spdiags(zeros(N,K), (-K:-1), A);

% Solve L*YY = RHS  and U*XX = YY and calculate norm of differences with
% provided colution arrays.
YY = zeros(N,R);
XX = zeros(N,R);
dY = zeros(1,R);
dX = zeros(1,R);

for i = 1:R
    YY(:,i) = L \ RHS(:,i);
    XX(:,i) = U \ YY(:,i);
    dY(i) = norm(Y(:,i) - YY(:,i)) / norm(Y(:,i));
    dX(i) = norm(X(:,i) - XX(:,i)) / norm(X(:,i));
end
% XX
% YY
% L = full(L)
% U = full(U)
% LU = full(A)

% for i =1:n
%     LU = [zeros(1,(2*k+1) - () )]
% end 

[v_max, i_max] = max(dY);
fprintf('Max. error after fwd. elim.: %g (for RHS # %i)\n', v_max, i_max);
[v_max, i_max] = max(dX);
fprintf('Max. error after bck. elim.: %g (for RHS # %i)\n', v_max, i_max);

    
