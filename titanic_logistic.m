Train = readtable('train.csv','Format','%f%f%f%q%C%f%f%f%q%f%q%C');
Test = readtable('test.csv','Format','%f%f%q%C%f%f%f%q%f%q%C');


Train.Fare(Train.Fare == 0 ) = NaN; % replacing fare 0 with NaN
Test.Fare(Test.Fare == 0 ) = NaN;
vars = Train.Properties.VariableNames;

% replacing unknown age with the average age
avgAge = nanmean(Train.Age);
Train.Age(isnan(Train.Age)) = avgAge;
Test.Age(isnan(Test.Age)) = avgAge;

% % Filling unknown fare based on what class and using the averages per class
% fare = grpstats(Train(:,{'Pclass','Fare'}),'Pclass');   % get class average
% %disp(fare)
% for i = 1:height(fare) % for each |Pclass|
%     % apply the class average to missing values
%     Train.Fare(Train.Pclass == i & isnan(Train.Fare)) = fare.mean_Fare(i);
%     Test.Fare(Test.Pclass == i & isnan(Test.Fare)) = fare.mean_Fare(i);
% end

% for passengers with unknown port of embarkation we will use the most
% frequent port Southampton

% get most frequent value
freqVal = mode(Train.Embarked);

% apply it to missling value
Train.Embarked(isundefined(Train.Embarked)) = freqVal;
Test.Embarked(isundefined(Test.Embarked)) = freqVal;

% Converting string in Sex category to double ( 1 = female and 2 = male )
Train.Sex = double(Train.Sex); 
Test.Sex = double(Test.Sex);



% Training set
% for Pclass 1, male
Train.Fare(Train.Pclass == 1 & Train.Sex == 2 & Train.Embarked == 'S') =  fixfare(Train.Fare(Train.Pclass == 1 & Train.Sex == 2 & Train.Embarked == 'S'));
Train.Fare(Train.Pclass == 1 & Train.Sex == 2 & Train.Embarked == 'C') = fixfare( Train.Fare(Train.Pclass == 1 & Train.Sex == 2 & Train.Embarked == 'C') );
Train.Fare(Train.Pclass == 1 & Train.Sex == 2 & Train.Embarked == 'Q') = fixfare( Train.Fare(Train.Pclass == 1 & Train.Sex == 2 & Train.Embarked == 'Q') );

% for Pclass 1 female
Train.Fare(Train.Pclass == 1 & Train.Sex == 1 & Train.Embarked == 'S') = fixfare( Train.Fare(Train.Pclass == 1 & Train.Sex == 1 & Train.Embarked == 'S') );
Train.Fare(Train.Pclass == 1 & Train.Sex == 1 & Train.Embarked == 'C') = fixfare( Train.Fare(Train.Pclass == 1 & Train.Sex == 1 & Train.Embarked == 'C') );
Train.Fare(Train.Pclass == 1 & Train.Sex == 1 & Train.Embarked == 'Q') = fixfare( Train.Fare(Train.Pclass == 1 & Train.Sex == 1 & Train.Embarked == 'Q') );

% for Pclass 2, male
Train.Fare(Train.Pclass == 2 & Train.Sex == 2 & Train.Embarked == 'S') = fixfare( Train.Fare(Train.Pclass == 2 & Train.Sex == 2 & Train.Embarked == 'S') );
Train.Fare(Train.Pclass == 2 & Train.Sex == 2 & Train.Embarked == 'C') = fixfare( Train.Fare(Train.Pclass == 2 & Train.Sex == 2 & Train.Embarked == 'C') );
Train.Fare(Train.Pclass == 2 & Train.Sex == 2 & Train.Embarked == 'Q') = fixfare( Train.Fare(Train.Pclass == 2 & Train.Sex == 2 & Train.Embarked == 'Q') );

% for Pclass 2 female
Train.Fare(Train.Pclass == 2 & Train.Sex == 1 & Train.Embarked == 'S') = fixfare( Train.Fare(Train.Pclass == 2 & Train.Sex == 1 & Train.Embarked == 'S') );
Train.Fare(Train.Pclass == 2 & Train.Sex == 1 & Train.Embarked == 'C') = fixfare( Train.Fare(Train.Pclass == 2 & Train.Sex == 1 & Train.Embarked == 'C') );
Train.Fare(Train.Pclass == 2 & Train.Sex == 1 & Train.Embarked == 'Q') = fixfare( Train.Fare(Train.Pclass == 2 & Train.Sex == 1 & Train.Embarked == 'Q') );

% for Pclass 3, male
Train.Fare(Train.Pclass == 3 & Train.Sex == 2 & Train.Embarked == 'S') = fixfare( Train.Fare(Train.Pclass == 3 & Train.Sex == 2 & Train.Embarked == 'S') );
Train.Fare(Train.Pclass == 3 & Train.Sex == 2 & Train.Embarked == 'C') = fixfare( Train.Fare(Train.Pclass == 3 & Train.Sex == 2 & Train.Embarked == 'C') );
Train.Fare(Train.Pclass == 3 & Train.Sex == 2 & Train.Embarked == 'Q') = fixfare( Train.Fare(Train.Pclass == 3 & Train.Sex == 2 & Train.Embarked == 'Q') );

% for Pclass 3 female
Train.Fare(Train.Pclass == 3 & Train.Sex == 1 & Train.Embarked == 'S') = fixfare( Train.Fare(Train.Pclass == 3 & Train.Sex == 1 & Train.Embarked == 'S') );
Train.Fare(Train.Pclass == 3 & Train.Sex == 1 & Train.Embarked == 'C') = fixfare( Train.Fare(Train.Pclass == 3 & Train.Sex == 1 & Train.Embarked == 'C') );
Train.Fare(Train.Pclass == 3 & Train.Sex == 1 & Train.Embarked == 'Q') = fixfare( Train.Fare(Train.Pclass == 3 & Train.Sex == 1 & Train.Embarked == 'Q') );

% Testing Set
% for Pclass 1, male
Test.Fare(Test.Pclass == 1 & Test.Sex == 2 & Test.Embarked == 'S') = fixfare(Test.Fare(Test.Pclass == 1 & Test.Sex == 2 & Test.Embarked == 'S'));
Test.Fare(Test.Pclass == 1 & Test.Sex == 2 & Test.Embarked == 'C') = fixfare(Test.Fare(Test.Pclass == 1 & Test.Sex == 2 & Test.Embarked == 'C'));
Test.Fare(Test.Pclass == 1 & Test.Sex == 2 & Test.Embarked == 'Q') = fixfare(Test.Fare(Test.Pclass == 1 & Test.Sex == 2 & Test.Embarked == 'Q'));

% for Pclass 1 female
Test.Fare(Test.Pclass == 1 & Test.Sex == 1 & Test.Embarked == 'S') = fixfare(Test.Fare(Test.Pclass == 1 & Test.Sex == 1 & Test.Embarked == 'S'));
Test.Fare(Test.Pclass == 1 & Test.Sex == 1 & Test.Embarked == 'C') = fixfare(Test.Fare(Test.Pclass == 1 & Test.Sex == 1 & Test.Embarked == 'C'));
Test.Fare(Test.Pclass == 1 & Test.Sex == 1 & Test.Embarked == 'Q') = fixfare(Test.Fare(Test.Pclass == 1 & Test.Sex == 1 & Test.Embarked == 'Q'));

% for Pclass 2, male
Test.Fare(Test.Pclass == 2 & Test.Sex == 2 & Test.Embarked == 'S') = fixfare(Test.Fare(Test.Pclass == 2 & Test.Sex == 2 & Test.Embarked == 'S'));
Test.Fare(Test.Pclass == 2 & Test.Sex == 2 & Test.Embarked == 'C') = fixfare(Test.Fare(Test.Pclass == 2 & Test.Sex == 2 & Test.Embarked == 'C'));
Test.Fare(Test.Pclass == 2 & Test.Sex == 2 & Test.Embarked == 'Q') = fixfare(Test.Fare(Test.Pclass == 2 & Test.Sex == 2 & Test.Embarked == 'Q'));

% for Pclass 2 female
Test.Fare(Test.Pclass == 2 & Test.Sex == 1 & Test.Embarked == 'S') = fixfare(Test.Fare(Test.Pclass == 2 & Test.Sex == 1 & Test.Embarked == 'S'));
Test.Fare(Test.Pclass == 2 & Test.Sex == 1 & Test.Embarked == 'C') = fixfare(Test.Fare(Test.Pclass == 2 & Test.Sex == 1 & Test.Embarked == 'C'));
Test.Fare(Test.Pclass == 2 & Test.Sex == 1 & Test.Embarked == 'Q') = fixfare(Test.Fare(Test.Pclass == 2 & Test.Sex == 1 & Test.Embarked == 'Q'));

% for Pclass 3, male
Test.Fare(Test.Pclass == 3 & Test.Sex == 2 & Test.Embarked == 'S') = fixfare(Test.Fare(Test.Pclass == 3 & Test.Sex == 2 & Test.Embarked == 'S'));
Test.Fare(Test.Pclass == 3 & Test.Sex == 2 & Test.Embarked == 'C') = fixfare(Test.Fare(Test.Pclass == 3 & Test.Sex == 2 & Test.Embarked == 'C'));
Test.Fare(Test.Pclass == 3 & Test.Sex == 2 & Test.Embarked == 'Q') = fixfare(Test.Fare(Test.Pclass == 3 & Test.Sex == 2 & Test.Embarked == 'Q'));

% for Pclass 3 female
Test.Fare(Test.Pclass == 3 & Test.Sex == 1 & Test.Embarked == 'S') = fixfare(Test.Fare(Test.Pclass == 3 & Test.Sex == 1 & Test.Embarked == 'S'));
Test.Fare(Test.Pclass == 3 & Test.Sex == 1 & Test.Embarked == 'C') = fixfare(Test.Fare(Test.Pclass == 3 & Test.Sex == 1 & Test.Embarked == 'C'));
Test.Fare(Test.Pclass == 3 & Test.Sex == 1 & Test.Embarked == 'Q') = fixfare(Test.Fare(Test.Pclass == 3 & Test.Sex == 1 & Test.Embarked == 'Q'));




% convert the data type from categorical to double
Train.Embarked = double(Train.Embarked);
Test.Embarked = double(Test.Embarked);

% tokenize the text string by white space
train_cabins = cellfun(@strsplit, Train.Cabin, 'UniformOutput', false);
test_cabins = cellfun(@strsplit, Test.Cabin, 'UniformOutput', false);

% count the number of tokens
Train.nCabins = cellfun(@length, train_cabins);
Test.nCabins = cellfun(@length, test_cabins);

% deal with exceptions - only the first class people had multiple cabins
Train.nCabins(Train.Pclass ~= 1 & Train.nCabins > 1,:) = 1;
Test.nCabins(Test.Pclass ~= 1 & Test.nCabins > 1,:) = 1;

% if |Cabin| is empty, then |nCabins| should be 0
Train.nCabins(cellfun(@isempty, Train.Cabin)) = 0;
Test.nCabins(cellfun(@isempty, Test.Cabin)) = 0;

% group values into separate bins
Train.AgeGroup = double(discretize(Train.Age, [0:10:20 65 80], ...
    'categorical',{'child','teen','adult','senior'}));
Test.AgeGroup = double(discretize(Test.Age, [0:10:20 65 80], ...
    'categorical',{'child','teen','adult','senior'}));


% group values into separate bins
Train.FareRange = double(discretize(Train.Fare, [0:10:30, 50, 120], ...
    'categorical',{'<10','10-20','20-30','30-50','>100'}));
Test.FareRange = double(discretize(Test.Fare, [0:10:30, 50, 120], ...
    'categorical',{'<10','10-20','20-30','30-50','>100'}));


Train.Squaresex = Train.Sex .* Train.Sex ;
Test.Squaresex = Test.Sex .* Test.Sex;



% We will recreate a new variable with the given table and all the usefull
% values from the table

% First considering only 10 features
X = zeros(height(Train),10);
Y = zeros(height(Train),1);

% we will use the data from Table to fill these values. Next split this
% dataset into training data and cross validataion dataset ( 600 - 300)

X(:,1) = Train.Sex;
X(:,2) = Train.Pclass;
X(:,3) = Train.Fare;
X(:,4) = Train.Age;
X(:,5) = Train.Embarked;
X(:,6) = Train.SibSp;
X(:,7) = Train.Parch;
X(:,8) = Train.nCabins;
X(:,9) = featureNormalize(Train.AgeGroup);
%X(:,11) = featureNormalize(Train.FareRange);
X(:,10) = featureNormalize(Train.Squaresex);
%X(:,11) = featureNormalize(Train.cubesex);

Y(:,1) = Train.Survived;

% we know that fare and age are having high variations and they are not
% properly normalized. So we will use feature normalization and bring them
% down to proper format with respect  to other features

[X(:,3),~,~] = featureNormalize(X(:,3));
[X(:,4),~,~] = featureNormalize(X(:,4));

%X(:,11) = X(:,3) .* X(:,3); 




% Now we will first test gradient descent algorithm without regularization

X_train = X(1:600,:); % first 301 to X and Y  training
Y_train = Y(1:600,:); 
X_test = X(601:end,:); % remaining 50 to X and Y testing
Y_test = Y(601:end,:);

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

%% Gradient Descent with Regularization
N=5000;
lambda = 1;
learning_rates = [0.1]; % declaring the learning rates used
[~,lr] = size(learning_rates); 

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


X2 = zeros(height(Test),10);
Y2 = zeros(height(Test),2);

% we will use the data from Table to fill these values. Next split this
% dataset into training data and cross validataion dataset ( 600 - 300)

X2(:,1) = Test.Sex;
X2(:,2) = Test.Pclass;
X2(:,3) = featureNormalize(Test.Fare);
X2(:,4) = featureNormalize(Test.Age);
X2(:,5) = Test.Embarked;
X2(:,6) = Test.SibSp;
X2(:,7) = Test.Parch;
X2(:,8) = Test.nCabins;
X2(:,9) = featureNormalize(Test.AgeGroup);
%X2(:,11) =featureNormalize(Test.FareRange);
X2(:,10) = featureNormalize(Test.Squaresex);
%X2(:,11) = featureNormalize(Test.cubesex);
%X2(:,11) = X2(:,3) .* X2(:,3);

Y2(:,1) = Test.PassengerId;

[m2, n2] = size(X2);
X2 = [ones(m2, 1) X2];
r_theta
p = predict(r_theta,X2);
Y2(:,2) = p;
Survived = p;

PassengerId = Test.PassengerId;
submission = table(PassengerId,Survived);   % combine them into a table
%disp(submission(1:5,:))                     % preview the table
writetable(submission,'submission.csv')     % write to a CSV file


Output = readtable('trythis.csv','Format','%f%f');

finaly = Output.Survived;
accuracy = mean(double(finaly == p)) * 100

close all
%disp(Train(1:30,:))