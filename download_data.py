import pandas as pd
import os

def download_titanic_data():
    """Download Titanic dataset from GitHub"""
    
    # URL ของ Titanic dataset
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    
    try:
        print("📥 Downloading Titanic dataset...")
        df = pd.read_csv(url)
        
        print(f"✅ Dataset downloaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"📋 Columns: {list(df.columns)}")
        
        # สร้าง data folder ถ้ายังไม่มี
        os.makedirs("data", exist_ok=True)
        
        # บันทึกไฟล์
        df.to_csv("data/titanic.csv", index=False)
        print("💾 Saved to data/titanic.csv")
        
        # แสดงข้อมูลตัวอย่าง
        print("\n📊 First 5 rows:")
        print(df.head())
        
        print("\n📈 Target distribution:")
        if 'Survived' in df.columns:
            print(df['Survived'].value_counts())
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")

if __name__ == "__main__":
    download_titanic_data()
