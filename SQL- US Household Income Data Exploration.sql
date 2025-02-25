SELECT *
FROM us_project.us_household_income;

SELECT *
FROM us_household_income_statistics;

#Checking area of land and area of water in different cities.
SELECT State_Name, County, City, ALand, AWater
FROM us_project.us_household_income;

#Comparing different states based on area of land and water.
SELECT State_Name, SUM(ALand) AS ALand, SUM(AWater) AS AWater
FROM us_project.us_household_income
GROUP BY State_Name
ORDER BY ALand DESC;

SELECT State_Name, SUM(ALand) AS ALand, SUM(AWater) AS AWater
FROM us_project.us_household_income
GROUP BY State_Name
ORDER BY ALand DESC
LIMIT 10;

SELECT State_Name, SUM(ALand) AS ALand, SUM(AWater) AS AWater
FROM us_project.us_household_income
GROUP BY State_Name
ORDER BY 3 DESC
LIMIT 10;

#Joining two tables to get more insights
SELECT *
FROM us_project.us_household_income u
JOIN us_project.us_household_income_statistics us
	ON u.id=us.id
WHERE Mean<>0; 

SELECT u.State_Name, County, Type, `Primary`, Mean, Median
FROM us_project.us_household_income u
JOIN us_project.us_household_income_statistics us
	ON u.id=us.id
WHERE Mean<>0; 

#Checking State-wise average and median incomes
SELECT u.State_Name, ROUND(AVG(Mean),1) AS Mean, ROUND(AVG(Median),1) AS Median
FROM us_project.us_household_income u
JOIN us_project.us_household_income_statistics us
	ON u.id=us.id
WHERE Mean<>0
GROUP BY u.State_Name
ORDER BY 2 DESC; 

SELECT u.State_Name, ROUND(AVG(Mean),1) AS Mean, ROUND(AVG(Median),1) AS Median
FROM us_project.us_household_income u
JOIN us_project.us_household_income_statistics us
	ON u.id=us.id
WHERE Mean<>0
GROUP BY u.State_Name
ORDER BY 2 DESC
LIMIT 10;

SELECT u.State_Name, ROUND(AVG(Mean),1) AS Mean, ROUND(AVG(Median),1) AS Median
FROM us_project.us_household_income u
JOIN us_project.us_household_income_statistics us
	ON u.id=us.id
WHERE Mean<>0
GROUP BY u.State_Name
ORDER BY 3 DESC;

#Checking Type-wise average and median salaries
SELECT Type, COUNT(Type), ROUND(AVG(Mean),1) AS Mean, ROUND(AVG(Median),1) AS Median
FROM us_project.us_household_income u
JOIN us_project.us_household_income_statistics us
	ON u.id=us.id
WHERE Mean<>0
GROUP BY Type
ORDER BY 3 DESC;

#Exploring the community type as it has the least average income
SELECT *
FROM us_project.us_household_income u
JOIN us_project.us_household_income_statistics us
	ON u.id=us.id
WHERE Mean<>0
AND Type='Community';   

#Only checking for those types where records are significiant(>100)
SELECT Type, COUNT(Type), ROUND(AVG(Mean),1) AS Mean, ROUND(AVG(Median),1) AS Median
FROM us_project.us_household_income u
JOIN us_project.us_household_income_statistics us
	ON u.id=us.id
WHERE Mean<>0
GROUP BY Type
HAVING COUNT(Type)>100
ORDER BY 3 DESC;

#Checking for cities' average income and median income
SELECT u.State_Name, City, ROUND(AVG(Mean),1) AS Mean, ROUND(AVG(Median),1) AS Median
FROM us_project.us_household_income u
JOIN us_project.us_household_income_statistics us
	ON u.id=us.id
WHERE Mean<>0
GROUP BY u.State_Name, City
ORDER BY Mean DESC;

