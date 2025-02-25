SELECT *
FROM us_project.us_household_income;

SELECT *
FROM us_household_income_statistics;

#Checking area of land and area of water in different cities
SELECT State_Name, County, City, ALand, AWater
FROM us_project.us_household_income;
