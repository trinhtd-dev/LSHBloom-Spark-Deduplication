#!/bin/bash

# Setup script for Big Data Processing environment (Spark 3.5.0 & Python 3.6.8)
# Target OS: Rocky Linux 8 / CentOS 8
set -e

echo "Starting system setup..."

# 1. Install Java 11
echo "[1/4] Installing OpenJDK 11..."
sudo dnf install -y java-11-openjdk-devel > /dev/null 2>&1

# 2. Download and configure Apache Spark 3.5.0
echo "[2/4] Configuring Apache Spark 3.5.0..."
cd ~
if [ ! -d "spark" ]; then
    wget -q --show-progress https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
    tar -xzf spark-3.5.0-bin-hadoop3.tgz
    rm spark-3.5.0-bin-hadoop3.tgz
    mv spark-3.5.0-bin-hadoop3 spark
fi

# Set environment variables
if ! grep -q "SPARK_HOME" ~/.bashrc; then
    echo 'export SPARK_HOME=$HOME/spark' >> ~/.bashrc
    echo 'export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin' >> ~/.bashrc
    echo 'export PYSPARK_PYTHON=python3' >> ~/.bashrc
fi

# Apply Spark resource configurations
cp ~/spark/conf/spark-defaults.conf.template ~/spark/conf/spark-defaults.conf
cat << 'EOF' > ~/spark/conf/spark-defaults.conf
spark.master                     local[8]
spark.driver.memory              2g
spark.executor.memory            6g
spark.sql.shuffle.partitions     16
EOF

# 3. Install Python dependencies
echo "[3/4] Installing Python dependencies..."
python3 -m pip install --user --upgrade pip > /dev/null 2>&1
python3 -m pip install --user numpy==1.19.5 datasets==1.18.0 > /dev/null 2>&1

# 4. Generate and run data ingestion script
echo "[4/4] Ingesting peS2o dataset (50,000 records)..."
mkdir -p ~/project_data
cd ~/project_data

cat << 'EOF' > download_data.py
import json
import os
import warnings
from datasets import load_dataset

warnings.filterwarnings("ignore")

os.makedirs("data", exist_ok=True)
file_path = "data/pes2o_sample.jsonl"

dataset = load_dataset("allenai/peS2o", split="train", streaming=True)

with open(file_path, "w", encoding="utf-8") as f:
    for i, row in enumerate(dataset):
        if i >= 50000:
            break
        record = {"id": row.get("id"), "text": row.get("text")}
        f.write(json.dumps(record) + "\n")

print(f"Data saved to {file_path}")
EOF

python3 download_data.py

echo "Setup completed successfully."
echo "Please run 'source ~/.bashrc' to apply environment variables before starting PySpark."
