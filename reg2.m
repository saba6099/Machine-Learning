%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the second part
%  of the exercise which covers regularization with logistic regression.
%
%  You will need to complete the following functions in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

%data = load('ex2data2.txt');
%X = data(:, [1, 2]); y = data(:, 3);
X = [1 0.1 0.01 0.001 0.0001 0.00001 0.000001; 1 0.2 0.04 0.008 0.0016 0.00032 0.000064; 1 0.3 0.009 0.027 0.0081 0.00243 0.000729; 
1 0.4 0.16 0.064 0.0256 0.01024 0.004096; 1 0.5 0.25 0.125 0.0625 0.03125 0.015625; 1 0.6 0.36 0.216 0.1296 0.07776 0.046656; 1 0.7 0.49 0.343 0.2401 0.16807 0.117649;
1 0.8 0.64 0.512 0.4096 0.32768 0.262144; 1 0.9 0.81 0.729 0.6561 0.59049 0.531441; 1 1 1 1 1 1 1]

y = [sin(2*pi*0.1); sin(2*pi*0.2); sin(2*pi*0.3); sin(2*pi*0.4); sin(2*pi*0.5); sin(2*pi*0.6); sin(2*pi*0.7); sin(2*pi*0.8); sin(2*pi*0.9); sin(2*pi*1)]
plot(X, y);

% Put some labels
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;


%% =========== Part 1: Regularized Logistic Regression ============
%  In this part, you are given a dataset with data points that are not
%  linearly separable. However, you would still like to use logistic
%  regression to classify the data points.
%
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
%

% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 0;
test_theta = ones(size(X,2),1);
% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(test_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros) at lambda = 0: %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:10));
fprintf('Norm %f', norm(grad));
%fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Compute and display cost and gradient
% with all-ones theta and lambda = 10

[cost, grad] = costFunctionReg(test_theta, X, y, 18);

fprintf('\nCost at test theta (with lambda = 18): %f\n', cost);
fprintf('Expected cost (approx): 3.16\n');
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:10));
fprintf('Norm %f', norm(grad));
%fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

[cost, grad] = costFunctionReg(test_theta, X, y, -18);

fprintf('\nCost at test theta (with lambda = -18): %f\n', cost);
fprintf('Expected cost (approx): 3.16\n');
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:10));
fprintf('Norm %f', norm(grad));
%fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

[cost, grad] = costFunctionReg(test_theta, X, y, 10000);

fprintf('\nCost at test theta (with lambda = 10000): %f\n', cost);
fprintf('Expected cost (approx): 3.16\n');
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:10));
fprintf('Norm %f', norm(grad));
%fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


[cost, grad] = costFunctionReg(test_theta, X, y, -10000);

fprintf('\nCost at test theta (with lambda = -10000): %f\n', cost);
fprintf('Expected cost (approx): 3.16\n');
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:10));
fprintf('Norm %f', norm(grad));
%fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%% ============= Part 2: Regularization and Accuracies =============
%  Optional Exercise:
%  In this part, you will get to try different values of lambda and
%  see how regularization affects the decision coundart
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?
%

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');

