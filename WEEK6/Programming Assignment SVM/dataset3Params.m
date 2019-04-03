function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_values = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_values = [0.01 0.03 0.1 0.3 1 3 10 30]';

training_error = zeros(length(C_values), length(sigma_values));
triple_error_c_sigma = zeros(length(C_values) + length(sigma_values), 3);
triple_idx = 1;

for i = 1:length(C_values)
    for j = 1: length(sigma_values)
        C_test = C_values(i);
        sigma_test = sigma_values(j);
          
        model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
        
        predictions = svmPredict(model, Xval);
        training_error(i, j) = mean(double(predictions ~= yval));
          
        triple_error_c_sigma(triple_idx, :) = [training_error(i, j), C_test, sigma_test];
        
        triple_idx = triple_idx + 1;
    end
end
  
sorted_triple_error_c_sigma = sortrows(triple_error_c_sigma, 1);
  
C = sorted_triple_error_c_sigma(1, 2);
sigma = sorted_triple_error_c_sigma(1, 3);

% =========================================================================

end
