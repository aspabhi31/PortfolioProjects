SELECT * FROM PortfolioProject.dbo.NashvilleHousing
--Standardize Date format
ALTER TABLE NashvilleHousing
ADD SaleDateConverted Date;
UPDATE NashvilleHousing
SET SaleDateConverted=CONVERT(Date,SaleDate)
SELECT SaleDateConverted FROM PortfolioProject.dbo.NashvilleHousing
--Populate Property Address data
SELECT * FROM PortfolioProject.dbo.NashvilleHousing WHERE PropertyAddress IS NULL
SELECT * FROM PortfolioProject.dbo.NashvilleHousing ORDER BY ParcelID
SELECT a.ParcelID, a.PropertyAddress, b.ParcelID, b.PropertyAddress, ISNULL(a.PropertyAddress, b.PropertyAddress) 
FROM PortfolioProject.dbo.NashvilleHousing a 
JOIN PortfolioProject.dbo.NashvilleHousing b 
ON a.ParcelID=b.ParcelID 
AND a.[UniqueID ]<>b.[UniqueID ]  
WHERE a.PropertyAddress IS NULL
UPDATE a
SET PropertyAddress=ISNULL(a.PropertyAddress, b.PropertyAddress)
FROM PortfolioProject.dbo.NashvilleHousing a
JOIN PortfolioProject.dbo.NashvilleHousing b
ON a.ParcelID=b.ParcelID
AND a.[UniqueID ]<>b.[UniqueID ]
--Breaking out Address into Individual Columns(Address, City, State)
SELECT PropertyAddress FROM PortfolioProject.dbo.NashvilleHousing
SELECT SUBSTRING(PropertyAddress, 1, CHARINDEX(',',PropertyAddress)-1) AS Address, SUBSTRING(PropertyAddress, CHARINDEX(',',PropertyAddress)+1, LEN(PropertyAddress)) AS CITY FROM PortfolioProject.dbo.NashvilleHousing 
ALTER TABLE NashvilleHousing
ADD PropertySplitAddress Nvarchar(255), PropertySplitCity Nvarchar(255)
SELECT * FROM PortfolioProject.dbo.NashvilleHousing
UPDATE NashvilleHousing 
SET PropertySplitAddress=SUBSTRING(PropertyAddress, 1, CHARINDEX(',',PropertyAddress)-1)
UPDATE NashvilleHousing
SET PropertySplitCity=SUBSTRING(PropertyAddress, CHARINDEX(',',PropertyAddress)+1, LEN(PropertyAddress))
SELECT * FROM PortfolioProject.dbo.NashvilleHousing

SELECT OwnerAddress FROM PortfolioProject.dbo.NashvilleHousing
SELECT PARSENAME(REPLACE(OwnerAddress, ',', '.'), 3) AS Address, PARSENAME(REPLACE(OwnerAddress, ',', '.'), 2) AS City, PARSENAME(REPLACE(OwnerAddress, ',', '.'), 1) AS State FROM PortfolioProject.dbo.NashvilleHousing 
ALTER TABLE NashvilleHousing
ADD OwnerSplitAddress Nvarchar(255), OwnerSplitCity Nvarchar(255), OwnerSplitState Nvarchar(255)
UPDATE NashvilleHousing
SET OwnerSplitAddress=PARSENAME(REPLACE(OwnerAddress, ',', '.'), 3)
UPDATE NashvilleHousing
SET OwnerSplitCity=PARSENAME(REPLACE(OwnerAddress, ',', '.'), 2)
UPDATE NashvilleHousing
SET OwnerSplitState=PARSENAME(REPLACE(OwnerAddress, ',', '.'), 1)
SELECT * FROM PortfolioProject.dbo.NashvilleHousing
--Change Y and N to Yes and No respectively in the 'SoldAsVacant' field
SELECT DISTINCT(SoldAsVacant) FROM PortfolioProject.dbo.NashvilleHousing
SELECT DISTINCT(SoldAsVacant), COUNT(SoldAsVacant) FROM PortfolioProject.dbo.NashvilleHousing GROUP BY SoldAsVacant ORDER BY 2
SELECT SoldAsVacant, 
CASE 
WHEN SoldAsVacant='Y' THEN 'YeS'
WHEN SoldAsvacant='N' THEN 'No'
ELSE SoldAsVacant
END AS SoldAsVacantFixed
FROM PortfolioProject.dbo.NashvilleHousing
SELECT SoldAsVacant, 
CASE 
WHEN SoldAsVacant='Y' THEN 'YeS'
WHEN SoldAsvacant='N' THEN 'No'
ELSE SoldAsVacant
END AS SoldAsVacantFixed
FROM PortfolioProject.dbo.NashvilleHousing 
WHERE SoldAsVacant='Y' OR SoldAsVacant='N'
UPDATE NashvilleHousing
SET SoldAsVacant=CASE 
WHEN SoldAsVacant='Y' THEN 'YeS'
WHEN SoldAsvacant='N' THEN 'No'
ELSE SoldAsVacant
END 
--Remove duplicates
SELECT *,
	ROW_NUMBER() OVER (
	PARTITION BY ParcelID,
				 PropertyAddress,
				 SalePrice,
				 SaleDate,
				 LegalReference
				 ORDER BY
					UniqueID
					) row_num
From PortfolioProject.dbo.NashvilleHousing
WITH RowNumCTE AS(
SELECT *,
	ROW_NUMBER() OVER (
	PARTITION BY ParcelID,
				 PropertyAddress,
				 SalePrice,
				 SaleDate,
				 LegalReference
				 ORDER BY
					UniqueID
					) row_num
From PortfolioProject.dbo.NashvilleHousing
)
Select *
From RowNumCTE
Where row_num > 1
WITH RowNumCTE AS(
SELECT *,
	ROW_NUMBER() OVER (
	PARTITION BY ParcelID,
				 PropertyAddress,
				 SalePrice,
				 SaleDate,
				 LegalReference
				 ORDER BY
					UniqueID
					) row_num
From PortfolioProject.dbo.NashvilleHousing
)
DELETE
From RowNumCTE
Where row_num > 1


--Delete unused columns
SELECT * FROM PortfolioProject.dbo.NashvilleHousing
ALTER TABLE NashvilleHousing
DROP COLUMN PropertyAddress, SaleDate, OwnerAddress, TaxDistrict