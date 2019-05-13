clear all;
load ionosphere;

Y1 = zeros(size(Y)); % Changing the g and b in y to 1 and 0 resp. 
for i=1:size(Y)
    if Y(i)== "g"
        Y1(i) = 1;
    else
        Y1(i) = 0;
    end
end
[data_size,~] = size(X); % size of the dataset
rng(1); % initializing a random permutation 
perm = randperm(data_size); % generating random dataset
X = X(perm,:); % shuffling the dataset
Y = Y(perm,:);

X_train = X(1:301,:); % first 301 to X and Y  training
Y_train = Y1(1:301,:); 
X_test = X(302:end,:); % remaining 50 to X and Y testing
Y_test = Y1(302:end,:);

%% Gradient Descent Algorithm without regularization
%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X_train);
[mt,~] = size(X_test);
% Add intercept term to x and X_test
X_train = [ones(m, 1) X_train];
X_test = [ones(mt, 1) X_test];
% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient

N = 4000;  % Number of iterations

learning_rates = [5,1,0.1,0.01,0.001]; % declaring the learning rates used
[~,lr] = size(learning_rates); 

train_error = zeros(N,lr);
test_error = zeros(N,lr);

for j=1:lr
    learning_rate = learning_rates(j); % iterating through each learning rate
    theta = initial_theta; % initializing the theta with zeros
    for i= 1:N
        [cost_train, grad] = LR_GD(theta, X_train, Y_train); % compute training MSE and gradient
        
        theta = theta - learning_rate * grad; % adjust the theta
        [cost_test, ~] = LR_GD(theta, X_test, Y_test); % compute the testing MSE
        train_error(i,j) = cost_train; % store the training MSE 
        test_error(i,j) = cost_test; % store the testing MSE 
    end
    % plotting the figures for different alpha with training and testing in
    % same diagram
    figure(j)
    plot(train_error(:,j))
    hold on;
    plot(test_error(:,j))
    hold off;
    titl = sprintf("Training and testing error for Learning rate=%d",learning_rate);
    title(titl)
    xlabel("Number of iterations")
    ylabel("Error")
    legend("Training Error","Testing Error")
end

%% Stochastic gradient descent algorithm without regularization

sto_train_error = zeros(N,lr);
sto_test_error = zeros(N,lr);

for j=1:lr
    learning_rate = learning_rates(j);
    theta = initial_theta;
    for i= 1:N
        for k=1:m % for stochastic gradient sending one training example at a time
            Xt = X_train(k,:); % selecting one training example
            Yt = Y_train(k,:);
            [cost_train, grad] = LR_GD(theta, Xt, Yt); % the same cost function with one example
            theta = theta - learning_rate * grad; 
        end
        [cost_test, ~] = LR_GD(theta, X_test, Y_test); % computing the test error 
        
        sto_train_error(i,j) = cost_train;
        sto_test_error(i,j) = cost_test;
    end
    figure(j+10)
    plot(sto_train_error(:,j))
    hold on;
    plot(sto_test_error(:,j))
    hold off;
    titl = sprintf("Stochastic Training and testing error for Learning rate=%d",learning_rate);
    title(titl)
    xlabel("Number of iterations")
    ylabel("Error")
    legend("Training Error","Testing Error")
end


%% Gradient Descent Algorithm with regularization
N= 5000;
% for lambda = 1
lambda = 1;

reg_training_cost = zeros(N,lr);
reg_test_cost = zeros(N,lr);

for j=1:lr
    learning_rate = learning_rates(j);
    r_theta = initial_theta;
    for i= 1:N

        [cost, grad] = LR_GDR(r_theta, X_train, Y_train, lambda);
        r_theta = r_theta - learning_rate * grad;
        reg_training_cost(i,j) = cost;
        [cost_test, ~] = LR_GDR(r_theta, X_test, Y_test,lambda);
        reg_test_cost(i,j) = cost_test;
    end
    figure(j+20)
    plot(reg_training_cost(:,j))
    hold on;
    plot(reg_test_cost(:,j))
    hold off;
    titl = sprintf("Regularized Training and testing error for Learning rate=%d and Lambda = %d",learning_rate,lambda);
    title(titl)
    xlabel("Number of iterations")
    ylabel("Error")
    legend("Training Error","Testing Error")
end

lambda = 0.1;

reg_training_cost = zeros(N,lr);
reg_test_cost = zeros(N,lr);

for j=1:lr
    learning_rate = learning_rates(j);
    r_theta = initial_theta;
    for i= 1:N

        [cost, grad] = LR_GDR(r_theta, X_train, Y_train, lambda);
        r_theta = r_theta - learning_rate * grad;
        reg_training_cost(i,j) = cost;
        [cost_test, ~] = LR_GDR(r_theta, X_test, Y_test,lambda);
        reg_test_cost(i,j) = cost_test;
    end
    figure(j+30)
    plot(reg_training_cost(:,j))
    hold on;
    plot(reg_test_cost(:,j))
    hold off;
    titl = sprintf("Regularized Training and testing error for Learning rate=%d and Lambda = %d",learning_rate,lambda);
    title(titl)
    xlabel("Number of iterations")
    ylabel("Error")
    legend("Training Error","Testing Error")
end


lambda = 5;

reg_training_cost = zeros(N,lr);
reg_test_cost = zeros(N,lr);

for j=1:lr
    learning_rate = learning_rates(j);
    r_theta = initial_theta;
    for i= 1:N

        [cost, grad] = LR_GDR(r_theta, X_train, Y_train, lambda);
        r_theta = r_theta - learning_rate * grad;
        reg_training_cost(i,j) = cost;
        [cost_test, ~] = LR_GDR(r_theta, X_test, Y_test,lambda);
        reg_test_cost(i,j) = cost_test;
    end
    figure(j+40)
    plot(reg_training_cost(:,j))
    hold on;
    plot(reg_test_cost(:,j))
    hold off;
    titl = sprintf("Regularized Training and testing error for Learning rate=%d and Lambda = %d",learning_rate,lambda);
    title(titl)
    xlabel("Number of iterations")
    ylabel("Error")
    legend("Training Error","Testing Error")
end








