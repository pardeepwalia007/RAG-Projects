## Business Metrics &amp; Calculation Definitions

## 1. Total Revenue

Definition:	Gross	sales	value	across	all	transactions.	Calculation:	SUM(revenue).	Used	to measure	business	scale,	not	profitability.

## 2. Total Orders

Definition:	Count	of	distinct	transactions.	Calculation:	COUNT(DISTINCT	sale_id). Represents	demand	volume.

## 3. Total Units Sold

Definition:	Volume	of	products	sold.	Calculation:	SUM(quantity).	Helps	identify	product movement	patterns.

## 4. Average Order Value (AOV)

Definition:	Average	revenue	per	order.	Calculation:	Total	Revenue	/	Total	Orders.	Sensitive to	pricing	and	product	mix.

## 5. Monthly Revenue

Definition:	Revenue	aggregated	by	month.	Calculation:	SUM(revenue)	GROUP	BY sale_year_month.	Used	for	trend	analysis.

## 6. Monthly Orders

Definition:	Orders	aggregated	by	month.	Calculation:	COUNT(DISTINCT	sale_id)	GROUP	BY sale_year_month.	Separates	demand	from	pricing	effects.

## 7. Category-Level Revenue

Definition:	Revenue	aggregated	by	product	category.	Calculation:	SUM(revenue)	GROUP	BY Category_ID.	Identifies	category	dependency.

## 8. Metric Usage Constraints

Metrics	must	not	be	used	to	infer	customer	behavior,	marketing	ROI,	or	profitability without	supporting	data.