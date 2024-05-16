function [beta, intercept, R2, MSE] = my_linreg(X, y)

    if length(X) ~= length(y)
        error('X and y must have the same number of elements.');
    end
    

    X = [ones(length(X),1) X(:)];
    

    beta = X \ y(:);
    

    intercept = beta(1);
    

    y_pred = X * beta;
    

    RSS = sum((y - y_pred).^2);
    TSS = sum((y - mean(y)).^2);
    R2 = 1 - RSS/TSS;
    MSE = RSS / (length(y) - size(X,2));
end

%%%%%%%%%%%%%%%%%%% How to use %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [beta, intercept, R2, MSE] = my_linreg(Type1(4,:)',y_infer(4,:)');
% 
% disp(['Regression coefficients: ', num2str(beta')]);
% disp(['Intercept: ', num2str(intercept)]);
% disp(['R-squared: ', num2str(R2)]);
% disp(['Mean Squared Error: ', num2str(MSE)]);