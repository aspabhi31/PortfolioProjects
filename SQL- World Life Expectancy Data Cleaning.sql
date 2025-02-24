SELECT * 
FROM world_life_expectancy;

#Query to check if there are duplicates in a data 
SELECT Country, Year, CONCAT(Country, Year), COUNT(CONCAT(Country, Year))
FROM world_life_expectancy
GROUP BY Country, Year, CONCAT(Country, Year)
HAVING COUNT(CONCAT(Country, Year))>1;

#Using Window functions to give row number to row ids to find the duplicate data
SELECT Row_ID,
CONCAT(Country, Year),
ROW_NUMBER() OVER(PARTITION BY CONCAT(Country, Year) ORDER BY CONCAT(Country, Year)) AS Row_Num
FROM world_life_expectancy;

#We get row ids where there is repetiton of data by the following query. We need to delete these row ids
SELECT *
FROM (
	SELECT Row_ID,
	CONCAT(Country, Year),
	ROW_NUMBER() OVER(PARTITION BY CONCAT(Country, Year) ORDER BY CONCAT(Country, Year)) AS Row_Num
	FROM world_life_expectancy
     ) AS Row_table
WHERE Row_Num>1;

DELETE FROM world_life_expectancy
WHERE Row_ID IN (
				SELECT Row_ID	
				FROM (
					  SELECT Row_ID,
					  CONCAT(Country, Year),
	                  ROW_NUMBER() OVER(PARTITION BY CONCAT(Country, Year) ORDER BY CONCAT(Country, Year)) AS Row_Num
	                  FROM world_life_expectancy
                      ) AS Row_table
                      WHERE Row_Num>1
                      );

#After removing duplicates, check how many blanks are there in the status column in data
SELECT *
FROM world_life_expectancy
WHERE Status='';

#Check how many distinct statuses are there in data
SELECT DISTINCT(Status)
FROM world_life_expectancy
WHERE Status<>'';

#Check which countries are developing and developed
SELECT DISTINCT(Country)
FROM world_life_expectancy
WHERE Status='Developing';

SELECT DISTINCT(Country)
FROM world_life_expectancy
WHERE Status='Developed';

#Update the blank statuses in the table, we will populate them by statuses of countries(which is constant in all the years)
UPDATE world_life_expectancy t1
JOIN world_life_expectancy t2
	ON t1.Country=t2.Country
SET t1.Status='Developing'
WHERE t1.Status=''
	AND t2.Status<>''
    AND t2.Status='Developing';
    
UPDATE world_life_expectancy t1
JOIN world_life_expectancy t2
	ON t1.Country=t2.Country
SET t1.Status='Developed'
WHERE t1.Status=''
	AND t2.Status<>''
    AND t2.Status='Developed';
    
#Check blank values in Life Expectancy column
SELECT *
FROM world_life_expectancy
WHERE `Life expectancy`='';  

SELECT Country, Year, `Life expectancy`
FROM world_life_expectancy;

#We will use INNER JOIN the following way, and will populate the blank life expectancy by taking the mean of adjacent two life expectancies
SELECT t1.Country, t1.Year, t1.`Life expectancy`,
t2.Country, t2.Year, t2.`Life expectancy`,
t3.Country, t3.Year, t3.`Life expectancy`,
ROUND((t2.`Life expectancy`+t3.`Life expectancy`)/2,1)
FROM world_life_expectancy t1
JOIN world_life_expectancy t2
	ON t1.Country=t2.Country
    AND t1.Year=t2.Year-1
JOIN world_life_expectancy t3
	ON t1.Country=t3.Country
	AND t1.Year=t3.Year+1
WHERE t1.`Life expectancy`='';

UPDATE world_life_expectancy t1
JOIN world_life_expectancy t2
	ON t1.Country=t2.Country
    AND t1.Year=t2.Year-1
JOIN world_life_expectancy t3
	ON t1.Country=t3.Country
	AND t1.Year=t3.Year+1
SET t1.`Life Expectancy`=ROUND((t2.`Life expectancy`+t3.`Life expectancy`)/2,1)
WHERE t1.`Life expectancy`='';    