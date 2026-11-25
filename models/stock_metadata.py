"""models/stock_metadata.py: hedge research implementation.

Functions are documented at their point of definition.
Historical outputs are research estimates, not executable orders.
"""

from dataclasses import dataclass

@dataclass
class StockMetadata:
    """Represent StockMetadata within the hedge research workflow.
    """
    name: str
    country: str
    exchange: str
    economic_sector: str
    industry_group: str
    industry: str
    currency: str | None = None