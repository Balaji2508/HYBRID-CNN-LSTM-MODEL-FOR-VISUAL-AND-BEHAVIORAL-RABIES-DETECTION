import pandas as pd
import os

# Config
OUTPUT_FILE = r"D:\balaji v2\Rabbies_Detection_Project\data\symptoms.csv"
NORMAL_TXT = r"D:\balaji v2\Rabbies_Detection_Project\data\text_data\normal_dogs.txt"
RABID_TXT = r"D:\balaji v2\Rabbies_Detection_Project\data\text_data\rabid_dogs.txt"

def generate_data():
    data = []
    
    # Read Rabid Data
    if os.path.exists(RABID_TXT):
        with open(RABID_TXT, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    data.append({"text": line.strip(), "label": "dumb_rabies"})
        print(f"Loaded {len(data)} lines from {RABID_TXT}")
    else:
        print(f"Warning: {RABID_TXT} not found.")

    # Read Normal Data
    normal_count = 0
    if os.path.exists(NORMAL_TXT):
        with open(NORMAL_TXT, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    data.append({"text": line.strip(), "label": "normal"})
                    normal_count += 1
        print(f"Loaded {normal_count} lines from {NORMAL_TXT}")
    else:
        print(f"Warning: {NORMAL_TXT} not found.")
            
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Generated {len(df)} samples at {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_data()
