function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

[rowmax,columnmax]=size(z);
for ii=1:rowmax
    for jj=1:columnmax
        g(ii,jj)=1/(1+exp(-z(ii,jj)));
    end
end


% =============================================================

end
