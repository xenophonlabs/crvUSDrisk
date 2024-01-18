"""
Provides the configs for simulating alternate
parameter sets. Parameters we test:

- Market Debt ceilings
- Loan and Liquidation Discounts
- Oracle chainlink limits
- PK Debt ceilings
- LLAMMA fees
"""

### ============ Debt Ceilings ============ ###

# pylint: disable=pointless-string-statement
"""
Methodology: being as conservative as possible, we can say that
(like) 99% of the debt ceiling is realized as debt. Of course, this
means that users would be paying really high rates, but let's pretend
that's reasonable.

It seems that, based on the current debt ceilings, things are ok. So
let's consider less conservative debt ceilings. We can then zoom in later.

We scale all of the debt ceilings by the same amount. We can then look
at the Bad Debt in each controller to see if different collaterals can handle
different increases.
"""

DEBT_CEILING_MULTIPLIER_SAMPLES = [2, 5, 10]

### ============ Loan and Liquidation Discounts ============ ###

# pylint: disable=pointless-string-statement
"""
Methodology: 
"""
