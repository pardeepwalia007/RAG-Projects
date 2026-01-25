## 📄 DOCUMENT: Coffee Shop Business Rules &amp; Constraints

## 1. METADATA &amp; DEFINITIONS

- ∞ Dataset Scope: Transaction logs for a coffee shop from March 2024 to March 2025.
- ∞ money: Represents Gross Revenue per transaction. Treat this as the primary "Sales" metric.
- ∞ coffee_name: Represents the Product.
- ∞ cash_type: Represents the Payment Method (Values: 'card', 'cash').
- ∞ card: Anonymized Customer Identifier. Use COUNT(DISTINCT card) to calculate "Unique Card-Paying Customers."

## 2. BUSINESS CALCULATION RULES

- ∞ Total Revenue: Calculated as SUM(money).
- ∞ Average Transaction Value (ATV): Calculated as AVG(money).
- ∞ Sales Volume: Calculated as COUNT(*) (number of cups sold).
- ∞ Payment Mix: Compare transactions where cash_type = 'card' vs cash_type = 'cash'.

## 3. STRICT OPERATIONAL CONSTRAINTS

- ∞ No Cost Analysis: The dataset does not contain "Cost" or "Profit" columns. DO NOT attempt to calculate Profit, Margin, or Net Revenue. If asked, state that cost data is missing.
- ∞ Customer Privacy: The card column contains masked IDs (e. g., ANON-0001 ). Never display a raw ID in an Executive Summary unless explicitly asked for "Customer ID debugging."
- ∞ Cash Customers: Transactions marked as cash_type = 'cash' have NULL card IDs. Do not treat these NULLs as a single unique customer; they are anonymous walk-ins.