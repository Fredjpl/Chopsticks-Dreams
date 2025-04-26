import fitz  # pymupdf
import logging
import sys
import re
import pandas as pd

# --- 设置日志 ---
logging.basicConfig(
    level=logging.INFO,  # 可以改成DEBUG看更细节的log
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

try:
    # --- 1. 打开PDF文件 ---
    pdf_path = 'document.pdf'
    logging.info(f"尝试打开PDF文件: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    if not doc:
        logging.error("PDF文件未能成功打开或为空。")
        raise FileNotFoundError(f"无法打开文件：{pdf_path}")
    
    logging.info(f"成功打开PDF文件，共 {len(doc)} 页。")

    # --- 2. 提取全文本 ---
    full_text = ""
    for i, page in enumerate(doc):
        page_text = page.get_text()
        full_text += page_text
        logging.debug(f"第 {i+1} 页提取完成，字符数：{len(page_text)}")

    if not full_text:
        logging.error("从PDF中未能提取到任何文本。")
        raise ValueError("PDF文本提取失败。")

    # --- 3. 定位'1.4.2 素菜'开始 ---
    start_idx = full_text.find("1.4.2 素菜")
    if start_idx == -1:
        logging.error("没有找到 '1.4.2 素菜' 开头。")
        raise ValueError("未找到'1.4.2 素菜'作为菜谱部分的开始。")

    menu_text = full_text[start_idx:]
    logging.info("'1.4.2 素菜'部分提取成功。")

    # --- 4. 定位结束点 ---
    end_idx = menu_text.find("1.5 进阶知识学习")  # 结束标志
    if end_idx != -1:
        menu_text = menu_text[:end_idx]
        logging.info("找到结束标志 '1.5 进阶知识学习'，截取菜谱部分。")
    else:
        logging.warning("没有找到结束标志 '1.5 进阶知识学习'，可能菜谱部分截取不完整。")

    # --- 5. 提取菜名 ---
    dishes = re.findall(r'[\u4e00-\u9fa5]{2,15}', menu_text)

    if not dishes:
        logging.error("菜名列表为空，提取失败。")
        raise ValueError("未提取到任何菜名，请检查PDF结构。")

    # --- 6. 打印提取结果 ---
    logging.info(f"成功提取到 {len(dishes)} 个菜名。")
    for dish in dishes:
        print(dish)

    # --- 7. 保存到Excel文件 ---
    output_excel = "dishes_list.xlsx"
    df = pd.DataFrame({"菜名": dishes})
    df.to_excel(output_excel, index=False)
    logging.info(f"菜名列表已成功保存到 {output_excel}")

except Exception as e:
    logging.exception(f"程序出错: {e}")


