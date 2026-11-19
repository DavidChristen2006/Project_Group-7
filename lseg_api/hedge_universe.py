import lseg.data as ld

def get_index_candidates(metadata):
    if metadata.country == "Germany":
        return [".GDAXI", ".STOXX50E", ".STOXX"]
    elif metadata.country == "France":
        return [ ".FCHI", ".STOXX50E", ".STOXX"]
    elif metadata.country in ["Italy", "Spain", "Netherlands"]:
        return [".STOXX50E", ".STOXX"]
    elif metadata.country == "Switzerland":
        return [".SSMI", ".SSHI"]
    else:
        return []

def get_broad_market_candidates(metadata):
    if metadata.country == "United States of America" and metadata.currency == "USD":
        return ["SPY", "VOO"]
    return []

def search_sector_etfs(metadata, max_results=30):
    results = ld.discovery.search(
        view=ld.discovery.Views.FUND_QUOTES,
        query=metadata.industry,
        filter="AssetCategory eq 'ETF'",
        select=("RIC, DocumentTitle, AssetCategory, ExchangeName"),
        top=max_results)
    if len(results) < 3:
        results = ld.discovery.search(
            view=ld.discovery.Views.FUND_QUOTES,
            query=metadata.economic_sector,
            filter="AssetCategory eq 'ETF'",
            select=("RIC, DocumentTitle, AssetCategory, ExchangeName"),
            top=max_results)
    return results

def filter_standard_etfs(etfs):
    if etfs.empty:
        return etfs
    data = etfs.copy()
    excluded_terms = ["2X", "3X", "Ultra", "Leveraged", "Inverse", "Short", "Bear"]
    pattern = "|".join(excluded_terms)
    data = data[
        ~data["DocumentTitle"]
        .str.contains(pattern, case=False, na=False)].copy()
    return data

def get_cross_hedge_universe(metadata, max_etf_results=20):
    index_candidates = get_index_candidates(metadata)
    broad_market_candidates = get_broad_market_candidates(metadata)
    sector_etfs = search_sector_etfs(metadata, max_results=max_etf_results)
    sector_etfs = filter_standard_etfs(sector_etfs)
    return {
        "indices": index_candidates,
        "broad_market": broad_market_candidates,
        "sector_etfs": sector_etfs}

def extract_etf_rics(hedge_universe, max_candidates=10):
    etfs = hedge_universe["sector_etfs"]
    if etfs.empty:
        return []
    return (
        etfs["RIC"]
        .dropna()
        .drop_duplicates()
        .head(max_candidates)
        .tolist())