SELECT * 
FROM world_life_expectancy;

#Check the life increase in all the countries.
SELECT Country,
MIN(`Life expectancy`),
MAX(`Life expectancy`),
ROUND(MAX(`Life expectancy`)-MIN(`Life expectancy`), 1) AS Life_Increase_15_Years
FROM world_life_expectancy
GROUP BY Country
HAVING MIN(`Life expectancy`)<>0
AND MAX(`Life expectancy`)<>0
ORDER BY Life_Increase_15_Years;

#Check average life expectancy of the world in different years.
SELECT Year,
ROUND(AVG(`Life expectancy`), 2) AS Average_Life_Expectancy
FROM world_life_expectancy
WHERE `Life expectancy`<>0
GROUP BY Year
ORDER BY Average_Life_Expectancy DESC;

#Check average life expectancy, average GDP of all countries. See if there is a correlation between GDP and Life expectancy.
SELECT Country,
ROUND(AVG(`Life expectancy`), 1) AS Life_Expectancy,
ROUND(AVG(GDP), 1) AS GDP
FROM world_life_expectancy
GROUP BY Country
HAVING Life_Expectancy>0 
AND GDP>0
ORDER BY GDP DESC;

# Check the number of low and high GDPs and the averaage life expectancies of high and low GDPs.
SELECT SUM(CASE WHEN GDP<=1500 THEN 1 ELSE 0 END) AS Low_GDP_Count,
AVG(CASE WHEN GDP<=1500 THEN `Life expectancy` ELSE NULL END) AS Low_GDP_Life_Expectancy,
SUM(CASE WHEN GDP>=1500 THEN 1 ELSE 0 END) AS High_GDP_Count,
AVG(CASE WHEN GDP>=1500 THEN `Life expectancy` ELSE NULL END) AS High_GDP_Life_Expectancy
FROM world_life_expectancy;

#Check the number of countries and average life expectancy in developed and developing statuses.
SELECT Status, ROUND(AVG(`Life expectancy`), 1) AS Life_Expectancy
FROM world_life_expectancy
GROUP BY Status;

SELECT Status, COUNT(DISTINCT Country) AS Number_Of_Countries
FROM world_life_expectancy
GROUP BY Status;

SELECT Status,
ROUND(AVG(`Life expectancy`), 1) AS Life_Expectancy, 
COUNT(DISTINCT Country) AS Number_Of_Countries
FROM world_life_expectancy
GROUP BY Status;

#Check the average life expectancy and BMI of all countries. Check if there is a correlation between the two.
SELECT Country, 
ROUND(AVG(`Life expectancy`), 1) AS Life_Expectancy,
ROUND(AVG(BMI),1) AS BMI
FROM world_life_expectancy
GROUP BY Country
HAVING Life_expectancy>0
AND BMI>0
ORDER BY BMI DESC;

#Check Adult Mortality in different countries in different years, also calculate rolling total to see the cumulative effect with years
SELECT Country,
Year,
`Life expectancy`,
`Adult Mortality`,
SUM(`Adult Mortality`) OVER(PARTITION BY Country ORDER BY Year) AS Rolling_Total
FROM world_life_expectancy;

#Also check for a specific country, say Canada
SELECT Country,
Year,
`Life expectancy`,
`Adult Mortality`,
SUM(`Adult Mortality`) OVER(PARTITION BY Country ORDER BY Year) AS Rolling_Total
FROM world_life_expectancy
WHERE Country LIKE '%Canada%';