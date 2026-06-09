"""
prep.py

2024-01-15T09:30:01 AAPL BUY 100 187.50
2024-01-15T09:30:02 MSFT SELL 50 410.20
2024-01-15T09:30:03 AAPL SELL 25 187.55
"""

def aggregate_trades(lines):
    results = dict()
    for line in lines:
        time, ticker, side, volume, price = line.split()
        volume = int(volume)
        price = float(price)

        if not ticker in results:
            results[ticker] = dict()
            results[ticker]["total_volume"] = 0
            results[ticker]["total_notional"] = 0.0
            results[ticker]["net_position"] = 0
            results[ticker]["net_cash"] = 0.0

        results[ticker]["total_volume"] += volume
        results[ticker]["total_notional"] += volume * price
        if side == "BUY":
            results[ticker]["net_position"] += volume
            results[ticker]["net_cash"] -= volume * price
        else:
            assert side == "SELL"
            results[ticker]["net_position"] -= volume
            results[ticker]["net_cash"] += volume * price

    return results


def main():
    lines = [
        "2024-01-15T09:30:01 AAPL BUY 100 187.50",
        "2024-01-15T09:30:02 MSFT SELL 50 410.20",
        "2024-01-15T09:30:03 AAPL SELL 25 187.55",
    ]
    results = aggregate_trades(lines)
    print(results)


if __name__ == "__main__":
    main()

