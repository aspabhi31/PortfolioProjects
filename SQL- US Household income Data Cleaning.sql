SELECT * 
FROM us_project.us_household_income;

SELECT * 
FROM us_project.us_household_income_statistics;

#Changing column name, id was written incorrectlly
ALTER TABLE us_household_income_statistics
RENAME COLUMN `ï»¿id` TO `id`;

SELECT COUNT(id)
FROM us_household_income;

SELECT COUNT(id)
FROM us_household_income_statistics;

#Checking if there are duplicates in a data
SELECT id, COUNT(id)
FROM us_household_income
GROUP BY id
HAVING COUNT(id)>1;

SELECT id, COUNT(id)
FROM us_household_income_statistics
GROUP BY id
HAVING COUNT(id)>1;

#Using Window functions to give row numbers to the ids, and identifying row ids where there is row number greater than 1
DELETE FROM us_household_income
WHERE row_id IN (
                 SELECT row_id  
                 FROM (
                       SELECT row_id, id, ROW_NUMBER() OVER(PARTITION BY id ORDER BY id) AS row_num 
                       FROM us_household_income
                       ) AS row_table
				 WHERE row_num>1
                );

#Standardize State name, checking state names to find the issue
SELECT State_Name, COUNT(State_Name)
FROM us_household_income
GROUP BY State_Name
ORDER BY 1; 

SELECT DISTINCT State_Name
FROM us_household_income
ORDER BY 1;

#Updating the table, to change the entries alabama and georgia to Alabama and Georgia respectively
UPDATE us_household_income
SET State_Name='Alabama'
WHERE State_Name='alabama';

UPDATE us_household_income
SET State_Name='Georgia'
WHERE State_Name='georia';

#Checking State abbreviations column
SELECT DISTINCT State_ab
FROM us_household_income
ORDER BY 1;

#Checking blank values in place
SELECT *
FROM us_household_income
WHERE Place=''; 

SELECT *
FROM us_household_income
WHERE County='Autauga County'; 

SELECT *
FROM us_household_income
WHERE County='Autauga County'
AND City='Vinemont'; 

#Populating the blank value, answer was already present in data
UPDATE us_household_income
SET Place='Autaugaville'
WHERE County='Autauga County'
AND City='Vinemont';

#Checking the column Type
SELECT Type, COUNT(Type)
FROM us_household_income
GROUP BY Type;

#We got two entries meaning the same thing, Boroughs and Borough. there were 128 entries of Borough and one entry of Boroughs
UPDATE us_household_income
SET Type='Borough'
WHERE Type='Boroughs';

#Checking the blank, null and 0 values in area of land and water
SELECT ALand, AWater
FROM us_household_income
WHERE (ALand=0 OR ALand='' OR ALand IS NULL)
OR (AWater=0 OR AWater='' OR AWater IS NULL);               