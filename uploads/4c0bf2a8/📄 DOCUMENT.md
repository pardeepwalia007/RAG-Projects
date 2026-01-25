## ***	COFFEE	SHOP	BUSINESS	DATA	RULES	***

## 1.	STRICT	SCHEMA	MAPPING	(READ	CAREFULLY)

- Revenue	-&gt;	Column	Name:	"money"
- Product	-&gt;	Column	Name:	"coffee_name"
- Payment	Method	-&gt;	Column	Name:	"cash_type"
- Customer	ID	-&gt;	Column	Name:	"card"

<!-- image -->

2.

## ⛔ FORBIDDEN	COLUMNS	(DO	NOT	HALLUCINATE)

The	following	columns	DO	NOT	EXIST	in	this	dataset.	Never	use	them	in	SQL:

<!-- image -->

- -❌ "quantity":	Does	not	exist.	You	MUST	use	COUNT(*)	to	calculate volume/units	sold.
- -❌ "sale_id":	Does	not	exist.	You	MUST	use	COUNT(*)	for	transaction	counts.
- -❌ "id":	Does	not	exist.
- -❌ "amount":	Does	not	exist.	Use	"money".
3. COMMON	METRIC	FORMULAS
- Total	Revenue:	SUM(money)
- Total	Volume	/	Units	Sold:	COUNT(*)
- Unique	Card	Customers:	COUNT(DISTINCT	card)	where	cash_type	=	'card'
- Average	Transaction	Value:	AVG(money)

<!-- image -->

<!-- image -->

<!-- image -->