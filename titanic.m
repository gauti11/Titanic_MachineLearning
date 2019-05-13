Train = readtable('train.csv','Format','%f%f%f%q%C%f%f%f%q%f%q%C');
Test = readtable('test.csv','Format','%f%f%q%C%f%f%f%q%f%q%C');
disp(Train(1:5,[2:3 5:8 10:11]))

Train.Fare(Train.Fare == 0 ) = NaN; % replacing fare 0 with NaN
Test.Fare(Test.Fare == 0 ) = NaN;
vars = Train.Properties.VariableNames;

% replacing unknown age with the average age
avgAge = nanmean(Train.Age);
Train.Age(isnan(Train.Age)) = avgAge;
Test.Age(isnan(Test.Age)) = avgAge;

% Filling unknown fare based on what class and using the averages per class
fare = grpstats(Train(:,{'Pclass','Fare'}),'Pclass');   % get class average
%disp(fare)
for i = 1:height(fare) % for each |Pclass|
    % apply the class average to missing values
    Train.Fare(Train.Pclass == i & isnan(Train.Fare)) = fare.mean_Fare(i);
    Test.Fare(Test.Pclass == i & isnan(Test.Fare)) = fare.mean_Fare(i);
end

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

% for passengers with unknown port of embarkation we will use the most
% frequent port Southampton

% get most frequent value
freqVal = mode(Train.Embarked);

% apply it to missling value
Train.Embarked(isundefined(Train.Embarked)) = freqVal;
Test.Embarked(isundefined(Test.Embarked)) = freqVal;

% convert the data type from categorical to double
Train.Embarked = double(Train.Embarked);
Test.Embarked = double(Test.Embarked);

% Converting string in Sex category to double
Train.Sex = double(Train.Sex);
Test.Sex = double(Test.Sex);

% The category Name, ticket and cabin have lot of unknown cells and the
% data is not usable
Train(:,{'Name','Ticket','Cabin'}) = [];
Test(:,{'Name','Ticket','Cabin'}) = [];

% Grouping the Age and Fare into categories

% group values into separate bins
Train.AgeGroup = double(discretize(Train.Age, [0:10:20 65 80], ...
    'categorical',{'child','teen','adult','senior'}));
Test.AgeGroup = double(discretize(Test.Age, [0:10:20 65 80], ...
    'categorical',{'child','teen','adult','senior'}));

% group values into separate bins
Train.FareRange = double(discretize(Train.Fare, [0:10:30, 100, 520], ...
    'categorical',{'<10','10-20','20-30','30-100','>100'}));
Test.FareRange = double(discretize(Test.Fare, [0:10:30, 100, 520], ...
    'categorical',{'<10','10-20','20-30','30-100','>100'}));

%yfit = predict(trainedClassifier, Test{:,trainedClassifier.predictFcn})

Y_train = Train.Survived;                   % slice response variable
X_train = Train(:,3:end);                   % select predictor variables
vars = X_train.Properties.VariableNames;    % get variable names
X_train = table2array(X_train);             % convert to a numeric matrix
X_test = table2array(Test(:,2:end));        % convert to a numeric matrix
categoricalPredictors = {'Pclass', 'Sex', 'Embarked', 'AgeGroup', 'FareRange'};
rng(1);                                     % for reproducibility
c = cvpartition(Y_train,'holdout', 0.30);   % 30%-holdout cross validation

% generate a Random Forest model from the partitioned data
RF = TreeBagger(200, X_train(training(c),:), Y_train(training(c)),...
    'PredictorNames', vars, 'Method','classification',...
    'CategoricalPredictors', categoricalPredictors, 'oobvarimp', 'on');

% compute the out-of-bag accuracy
oobAccuracy = 1 - oobError(RF, 'mode', 'ensemble')

[~,order] = sort(RF.OOBPermutedVarDeltaError);  % sort the metrics
figure
barh(RF.OOBPermutedVarDeltaError(order))        % horizontal bar chart
title('Feature Importance Metric')
ax = gca; ax.YTickLabel = vars(order);          % variable names as labels

[Yfit, Yscore] = predict(RF, X_train(test(c),:));       % use holdout data
cfm = confusionmat(Y_train(test(c)), str2double(Yfit)); % confusion matrix
cvAccuracy = sum(cfm(logical(eye(2))))/length(Yfit)     % compute accuracy

posClass = strcmp(RF.ClassNames,'1');   % get the index of the positive class
curves = zeros(2,1); labels = cell(2,1);% pre-allocated variables
[rocX, rocY, ~, auc] = perfcurve(Y_train(test(c)),Yscore(:,posClass),'1');
figure
curves(1) = plot(rocX, rocY);           % use the perfcurve output to plot
labels{1} = sprintf('Random Forest - AUC: %.1f%%', auc*100);
curves(end) = refline(1,0); set(curves(end),'Color','r');
labels{end} = 'Reference Line - A random classifier';
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC Plot')
legend(curves, labels, 'Location', 'SouthEast')


PassengerId = Test.PassengerId;             % extract Passenger Ids
Survived = predict(RF, X_test);             % generate response variable
Survived = str2double(Survived);            % convert to double
submission = table(PassengerId,Survived);   % combine them into a table
disp(submission(1:5,:))                     % preview the table
writetable(submission,'submission.csv')     % write to a CSV file