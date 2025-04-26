import pandas as pd
import logging
import sys
import time
from openai import AzureOpenAI

# --- 设置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- Azure OpenAI设置 ---
endpoint = "https://jiany-m9yk24aj-eastus2.cognitiveservices.azure.com/"
api_version = "2024-02-15-preview"  # 看你的部署版本，也可能是 2024-12-01-preview
api_key = "76C31SFkEbYQgMECT1f9YJSoqW6ZGb95pmSPVRo1xC02nqXScgNAJQQJ99BDACHYHv6XJ3w3AAAAACOGYACm"
deployment_name = "gpt-4o"  # 非常重要，和Portal上Deployment Name一模一样

# --- 建立客户端 ---
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_version=api_version,
    api_key=api_key
)

# --- 读取菜名列表 ---
input_excel = "菜名列表.xlsx"
output_excel = "菜名列表_带估算时间.xlsx"

try:
    df = pd.read_excel(input_excel)
    if "菜名" not in df.columns:
        raise ValueError("Excel中找不到'菜名'列。")
    logging.info(f"成功读取到 {len(df)} 道菜。")
except Exception as e:
    logging.exception(f"读取Excel失败: {e}")
    sys.exit(1)

# --- 给每道菜估时间 ---
estimated_times = []

for idx, row in df.iterrows():
    dish_name = row['菜名']
    try:
        prompt = f"""请根据你的常识，合理估算从开始准备到完成做完这道菜需要多少分钟（包括准备和烹饪时间）。请只回答一个整数加分钟单位，比如“25分钟”。
菜名：{dish_name}"""

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "你是一个专业的厨师助手，负责估算做菜时间。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0,
            top_p=1.0
        )
        time_text = response.choices[0].message.content.strip()
        estimated_times.append(time_text)
        logging.info(f"{dish_name} ➔ {time_text}")

        time.sleep(0.3)

    except Exception as e:
        logging.error(f"估算 {dish_name} 失败: {e}")
        logging.error(f"中止运行，请检查问题后重新开始。")
        # 估算失败时，保存当前进度
        df = df.iloc[:idx]  # 只保留到出错前的数据
        df["估算时间"] = estimated_times
        df.to_excel(output_excel, index=False)
        logging.info(f"保存到当前位置：{output_excel}")
        sys.exit(1)  # 直接中断程序

# --- 保存新Excel（如果全部跑完才保存）---
df["估算时间"] = estimated_times

try:
    df.to_excel(output_excel, index=False)
    logging.info(f"成功保存带估算时间的新文件：{output_excel}")
except Exception as e:
    logging.exception(f"保存Excel失败: {e}")

