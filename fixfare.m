function g = fixfare(thelist)

% First we will fix fare for all the Pclass 1 guys male and female w.r.t
% where they embarked C, S and Q
% starting with male ( where sex = 2)

%thelist = Train.Fare(Train.Pclass == 1 & Train.Sex == 2 & Train.Embarked == 'S');

mean = nanmean(thelist);
e = thelist/mean;
r = ( e > 1);
a = e .* r ;
g = zeros(size(a)) ;
for i=1:size(a)
    if a(i) ~= NaN && a(i) ~= 0
        g(i) = thelist(i)/a(i);
    else
        g(i) = thelist(i);
    end
end

new_mean = nanmean(g);
idx = isnan(g);
g(idx) = new_mean;


% testing
idx = isnan(thelist);
thelist(idx) = new_mean;

%g = thelist;

end