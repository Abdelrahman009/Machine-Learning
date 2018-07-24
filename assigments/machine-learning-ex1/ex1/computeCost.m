function J = computeCost(X, y, theta)

m = length(y); 
A = (X*theta - y);

J =  (A'*A)/2/m;
end
