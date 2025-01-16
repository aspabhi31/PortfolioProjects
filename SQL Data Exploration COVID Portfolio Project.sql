SELECT * FROM PortfolioProject..CovidDeaths WHERE continent IS NOT NULL ORDER BY 3,4
SELECT * FROM PortfolioProject..CovidVaccinations WHERE continent IS NOT NULL ORDER BY 3,4
--Select data that we are going to be using
SELECT location, date, total_cases, new_cases, total_deaths, population FROM PortfolioProject..CovidDeaths WHERE continent IS NOT NULL ORDER BY 1,2
--Looking at Total cases vs Total Deaths
--Shows likelihood of dying if you contract covid
SELECT location, date, total_cases, total_deaths, (total_deaths/total_cases)*100 AS DeathPercentage FROM PortfolioProject..CovidDeaths WHERE continent IS NOT NULL ORDER BY 1,2
SELECT location, date, total_cases, total_deaths, (total_deaths/total_cases)*100 AS DeathPercentage FROM PortfolioProject..CovidDeaths WHERE location LIKE 'C%anad%' ORDER BY 1,2
--Looking at total cases vs population
--Shows what percentage of population contracted covid
SELECT location, date, population, total_cases, (total_cases/population)*100 AS PopulationPercentage FROM PortfolioProject..CovidDeaths WHERE continent IS NOT NULL ORDER BY 1,2
SELECT location, date, population, total_cases, (total_cases/population)*100 AS PopulationPercentage FROM PortfolioProject..CovidDeaths WHERE location LIKE 'C%anad%' ORDER BY 1,2
--Looking at the Countries with highest infection rate
SELECT location, population, MAX(total_cases) AS HighestInfectionCount, MAX((total_cases/population))*100 AS PercentPopulationInfected FROM PortfolioProject..CovidDeaths WHERE continent IS NOT NULL GROUP BY location, population ORDER BY PercentPopulationInfected DESC
--Showing Countries with highest death count
SELECT location, MAX(CAST(total_deaths AS INT)) AS TotalDeaths FROM PortfolioProject..CovidDeaths WHERE continent IS NOT NULL GROUP BY location ORDER BY TotalDeaths DESC
--Showing Continents with highest death count
SELECT continent, MAX(CAST(total_deaths AS INT)) AS TotalDeaths FROM PortfolioProject..CovidDeaths WHERE continent IS NOT NULL GROUP BY continent ORDER BY TotalDeaths DESC
SELECT location, MAX(CAST(total_deaths AS INT)) AS TotalDeaths FROM PortfolioProject..CovidDeaths WHERE continent IS NULL GROUP BY location ORDER BY TotalDeaths DESC
--Global numbers
SELECT date, SUM(new_cases) AS TotalCases, SUM(CAST(new_deaths AS INT)) AS TotalDeaths, SUM(CAST(new_deaths AS INT))/(SUM(new_cases))*100 AS DeathPercentage FROM PortfolioProject..CovidDeaths WHERE continent IS NOT NULL GROUP BY date ORDER BY 1, 2
SELECT SUM(new_cases) AS TotalCases, SUM(CAST(new_deaths AS INT)) AS TotalDeaths, SUM(CAST(new_deaths AS INT))/(SUM(new_cases))*100 AS DeathPercentage FROM PortfolioProject..CovidDeaths WHERE continent IS NOT NULL ORDER BY 1, 2
--Looking at total population vs vaccinations
SELECT dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations, SUM(CONVERT(INT,vac.new_vaccinations)) OVER (PARTITION BY dea.location ORDER BY dea.location, dea.date) AS RollingPeopleVaccinated FROM PortfolioProject..CovidDeaths dea JOIN PortfolioProject..CovidVaccinations vac ON dea.location=vac.location AND dea.date=vac.date WHERE dea.continent IS NOT NULL ORDER BY 2, 3
--Using CTE
With PopVsVac(Continent, Location, Date, Population, NewVaccinations, RollingPeopleVaccinated) 
AS (SELECT dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
SUM(CONVERT(INT,vac.new_vaccinations)) OVER (PARTITION BY dea.location ORDER BY dea.location, dea.date) AS RollingPeopleVaccinated 
FROM PortfolioProject..CovidDeaths dea 
JOIN PortfolioProject..CovidVaccinations vac 
ON dea.location=vac.location AND dea.date=vac.date 
WHERE dea.continent IS NOT NULL)
SELECT location, MAX(RollingPeopleVaccinated/Population)*100 AS PercentPopulationVaccinated 
FROM PopVsVac 
GROUP BY location 
ORDER BY PercentPopulationVaccinated DESC
--Using Temp Table
DROP TABLE IF EXISTS #PercentPopulationVaccinated 
CREATE TABLE #PercentPopulationVaccinated
(Continent nvarchar(255),
 Location nvarchar(255),
 Date datetime,
 Population numeric,
 NewVaccinations numeric,
 RollingPeopleVaccinated numeric)
 INSERT INTO #PercentPopulationVaccinated
 SELECT dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations, 
 SUM(CONVERT(INT,vac.new_vaccinations)) OVER (PARTITION BY dea.location ORDER BY dea.location, dea.date) AS RollingPeopleVaccinated 
 FROM PortfolioProject..CovidDeaths dea JOIN PortfolioProject..CovidVaccinations vac 
 ON dea.location=vac.location AND dea.date=vac.date 
 WHERE dea.continent IS NOT NULL
 SELECT Location, MAX(RollingPeopleVaccinated/Population)*100 AS PercentPopulationVaccinated FROM #PercentPopulationVaccinated 
 GROUP BY location 
 ORDER BY PercentPopulationVaccinated DESC
 --Creating View to store data for later visualizations
 CREATE VIEW PercentPopulationVaccinated AS 
 SELECT dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations, 
 SUM(CONVERT(INT,vac.new_vaccinations)) OVER (PARTITION BY dea.location ORDER BY dea.location, dea.date) AS RollingPeopleVaccinated 
 FROM PortfolioProject..CovidDeaths dea 
 JOIN PortfolioProject..CovidVaccinations vac 
 ON dea.location=vac.location AND dea.date=vac.date 
 WHERE dea.continent IS NOT NULL 
 --Selecting Data from the View
 SELECT * FROM PercentPopulationVaccinated
 SELECT location, population, MAX(RollingPeopleVaccinated)/population*100 AS PercentVaccinated FROM PercentPopulationVaccinated GROUP BY location, population ORDER BY PercentVaccinated DESC

