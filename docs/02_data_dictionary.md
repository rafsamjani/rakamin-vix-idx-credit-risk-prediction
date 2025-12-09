# Data Dictionary

## Dataset Overview
- **Source**: Lending Club Loan Data (2007-2014)
- **Format**: CSV file with borrower information and loan outcomes
- **Records**: Historical loan applications and their final status
- **Time Period**: 2007-2014

## Key Variable Categories

### Loan Information
| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| `loan_amnt` | Numeric | The listed amount of the loan applied for by the borrower | $1,000 - $35,000 |
| `funded_amnt` | Numeric | The total amount committed to that loan at that point in time | $1,000 - $35,000 |
| `term` | Categorical | The number of payments on the loan | 36 months, 60 months |
| `int_rate` | Numeric | Interest rate on the loan | 5.32% - 26.06% |
| `installment` | Numeric | The monthly payment owed by the borrower | $15.69 - $1,408.79 |
| `grade` | Categorical | LC assigned loan grade | A, B, C, D, E, F, G |
| `sub_grade` | Categorical | LC assigned loan subgrade | A1-A5, B1-B5, ..., G1-G5 |

### Borrower Information
| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| `emp_title` | Text | The job title supplied by the borrower | Various |
| `emp_length` | Categorical | Employment length in years | < 1 year to 10+ years |
| `home_ownership` | Categorical | Home ownership status | RENT, OWN, MORTGAGE, OTHER |
| `annual_inc` | Numeric | Annual income reported by the borrower | $1,896 - $9,550,000 |
| `verification_status` | Categorical | Income verification status | Source Verified, Verified, Not Verified |
| `purpose` | Categorical | Category for loan purpose | debt_consolidation, credit_card, home_improvement, etc. |
| `title` | Text | Loan title provided by borrower | Various |

### Credit History
| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| `delinq_2yrs` | Numeric | Number of delinquencies in past 2 years | 0 - 39 |
| `earliest_cr_line` | Date | Date of earliest credit line | 1946-2014 |
| `fico_range_low` | Numeric | Lower boundary of FICO score | 660 - 845 |
| `fico_range_high` | Numeric | Upper boundary of FICO score | 665 - 850 |
| `inq_last_6mths` | Numeric | Number of inquiries in last 6 months | 0 - 33 |
| `mths_since_last_delinq` | Numeric | Months since last delinquency | 0 - 188 |
| `mths_since_last_record` | Numeric | Months since last public record | 0 - 120 |
| `open_acc` | Numeric | Number of open credit lines | 1 - 90 |
| `pub_rec` | Numeric | Number of derogatory public records | 0 - 86 |
| `revol_bal` | Numeric | Total credit revolving balance | $0 - $2,492,571 |
| `revol_util` | Numeric | Revolving line utilization rate | 0% - 100% |
| `total_acc` | Numeric | Total number of credit lines | 2 - 169 |

### Loan Status (Target Variable)
| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| `loan_status` | Categorical | Current status of the loan | **Fully Paid**, **Charged Off**, Current, Late (31-120 days), etc. |

### Application Information
| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| `issue_d` | Date | Month credit was issued | 2007-2014 |
| `application_type` | Categorical | Type of application | Individual, Joint |
| `addr_state` | Categorical | State of borrower | All 50 US states |
| `dti` | Numeric | Debt-to-income ratio | 0% - 39.99% |
| `annual_inc_joint` | Numeric | Combined annual income (for joint apps) | $0 - $9,550,000 |

## Target Variable Definition

### Binary Classification
For this project, we'll focus on completed loans:
- **Fully Paid**: Loan successfully repaid (Positive class: 0)
- **Charged Off**: Loan defaulted (Negative class: 1)

### Excluded Status Values
The following loan statuses will be excluded from modeling:
- `Current`: Still active
- `Late (31-120 days)`: Currently delinquent
- `In Grace Period`: Temporary status
- `Late (16-30 days)`: Minor delinquency
- `Default`: Technical default status

## Key Risk Indicators

### High Risk Factors
- Low FICO scores (< 650)
- High debt-to-income ratio (> 30%)
- Multiple recent inquiries
- Previous delinquencies
- Short credit history

### Low Risk Factors
- High FICO scores (> 750)
- Low debt-to-income ratio (< 15%)
- Long employment history
- Home ownership
- Low revolving utilization (< 20%)

## Data Quality Considerations

### Missing Values
Common variables with missing data:
- `emp_title`: Self-reported, often missing
- `emp_length`: May not be applicable
- `annual_inc_joint`: Only for joint applications
- `mths_since_last_delinq`: No prior delinquencies
- `revol_util`: May be missing for inactive credit

### Outliers
- `annual_inc`: Extremely high incomes may need capping
- `revol_bal`: Very large balances may indicate business use
- `total_acc`: Very high number of accounts may be unusual

### Feature Engineering Opportunities

### Derived Variables
1. **FICO Score Average**: `(fico_range_low + fico_range_high) / 2`
2. **Credit History Length**: Months from `earliest_cr_line` to `issue_d`
3. **Loan-to-Income Ratio**: `loan_amnt / annual_inc`
4. **Monthly Debt Burden**: `installment / (annual_inc / 12)`

### Categorical Encoding
1. **Grade Numeric**: Convert A-G to 1-7
2. **Emp Length Numeric**: Convert text to numeric years
3. **State Risk**: Group states by default rates
4. **Purpose Risk**: Group purposes by risk level

## Business Logic Rules

### Exclusion Criteria
- Annual income < $20,000 (minimum income requirement)
- Loan amount > 50% of annual income (high leverage)
- FICO score < 620 (below minimum threshold)
- Employment length < 6 months (unstable income)

### Risk Tiers
| Risk Tier | FICO Range | Expected Default Rate |
|-----------|-------------|----------------------|
| Excellent | 750+ | < 2% |
| Good | 700-749 | 2-5% |
| Average | 650-699 | 5-10% |
| Poor | 620-649 | 10-20% |
| Very Poor | < 620 | > 20% |

## Data Preprocessing Pipeline

### Cleaning Steps
1. **Filter**: Keep only Fully Paid and Charged Off loans
2. **Handle Missing**: Impute or drop variables with > 50% missing
3. **Outlier Treatment**: Cap extreme values at 99th percentile
4. **Format**: Convert dates, percentages, and categorical variables
5. **Feature Creation**: Generate derived risk indicators

### Transformation Steps
1. **Numeric**: Standardize or normalize continuous variables
2. **Categorical**: One-hot encode nominal variables
3. **Ordinal**: Label encode ordinal variables
4. **Target**: Handle class imbalance using SMOTE or weighting

## Model Input Features

### Final Feature Set
After preprocessing, the model will use approximately 20-30 features including:

**Core Features:**
- Loan characteristics (amount, term, interest rate)
- Borrower demographics (income, employment, home ownership)
- Credit history (FICO score, delinquencies, credit age)
- Debt ratios (DTI, revolving utilization)

**Engineered Features:**
- Risk indicators and ratios
- Historical payment patterns
- Geographic risk factors
- Purpose-specific risk scores

This comprehensive feature set will enable the model to capture both traditional credit risk factors and subtle patterns that indicate higher default probability.