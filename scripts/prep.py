"""
prep.py

2024-01-15T09:30:01 AAPL BUY 100 187.50
2024-01-15T09:30:02 MSFT SELL 50 410.20
2024-01-15T09:30:03 AAPL SELL 25 187.55
"""

def parse_trades(filename):
    results = dict()

    with open(filename) as infile:
        for line in infile:
            time, ticker, side, volume, price = line.split()
            volume = int(volume)
            price = float(price)

            if not ticker in results:
                results[ticker] = dict()
                results[ticker]["total_volume"] = 0.0
                results[ticker]["total_notional"] = 0.0

            results[ticker]["total_volume"] += volume
            results[ticker]["total_notional"] += volume * price

    return results


def main():
    filename = "example.txt"
    results = parse_trades(filename)
    print(results)


if __name__ == "__main__":
    main()

