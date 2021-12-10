mu = transpose([1,2,2]);
sigma = [[1,0,0], [0,5,2], [0,2,5]];
sigmaDet = det(sigma);
x_0 = transpose([0.5,0,1]);
exp_mahalanobis = exp(-0.5 * transpose(x_0 - mu) * sigma * (x_0 - mu));
exp_mahalanobis / sqrt((2*pi)^3 * sigmaDet);
