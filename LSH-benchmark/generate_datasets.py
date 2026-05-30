import argparse
import subprocess


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p-values",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated p values to generate",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="test_p_",
        help="Output folder prefix",
    )
    return parser.parse_args()


def parse_p_values(raw):
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def main():
    args = get_args()
    p_values = parse_p_values(args.p_values)

    print(f"{'=' * 20} GENERATING DATASETS {'=' * 20}")
    for p in p_values:
        benchmark_name = f"{args.output_prefix}{p}"
        print(f"Generating dataset {benchmark_name} (Duplicate proportion: {p})...")
        subprocess.run(
            [
                "python",
                "synthetic_benchmark/create_lshbloom_benchmark.py",
                "-p",
                str(p),
                "-o",
                benchmark_name,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
