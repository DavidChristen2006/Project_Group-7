from dataclasses import dataclass

@dataclass
class StockMetadata:
    name: str
    country: str
    exchange: str
    economic_sector: str
    industry_group: str
    industry: str
    currency: str | None = None