import streamlit as st
import pandas as pd
import datetime
from datetime import date
import requests
import json
import base64
from io import StringIO, BytesIO
import os
import pdfplumber
import re

# --- 页面配置 ---
st.set_page_config(page_title="AI 智能账本", page_icon="💰", layout="wide")

# --- 常量配置 ---
DEFAULT_TARGET_SPEND = 60.0  # 每日体面支出标准
GITHUB_API_URL = "https://api.github.com"
VISION_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct" 
TEXT_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"

# --- 存储类 ---
class DataManager:
    """数据管理类，支持 GitHub 远程存储和本地 CSV 存储"""
    def __init__(self, github_token=None, repo=None, filename="ledger.csv"):
        self.github_token = github_token
        if repo and repo.startswith("http"):
            self.repo = repo.rstrip("/").split("github.com/")[-1]
        else:
            self.repo = repo
        self.filename = filename
        self.use_github = bool(github_token and self.repo)

    def load_data(self):
        if self.use_github:
            return self._load_from_github()
        else:
            return self._load_from_local()

    def save_data(self, df, sha=None):
        if self.use_github:
            return self._save_to_github(df, sha)
        else:
            return self._save_to_local(df)

    def _load_from_local(self):
        if os.path.exists(self.filename):
            try:
                return pd.read_csv(self.filename), None
            except:
                return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"]), None
        return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"]), None

    def _save_to_local(self, df):
        df.to_csv(self.filename, index=False)
        return True

    def _load_from_github(self):
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        url = f"{GITHUB_API_URL}/repos/{self.repo}/contents/{self.filename}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            content = response.json()
            csv_str = base64.b64decode(content['content']).decode('utf-8')
            try:
                return pd.read_csv(StringIO(csv_str)), content['sha']
            except pd.errors.EmptyDataError:
                return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"]), content['sha']
        elif response.status_code == 404:
            return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"]), None
        else:
            st.error(f"GitHub 读取错误: {response.status_code}")
            return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"]), None

    def _save_to_github(self, df, sha):
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        csv_str = df.to_csv(index=False)
        content_bytes = base64.b64encode(csv_str.encode('utf-8')).decode('utf-8')
        
        url = f"{GITHUB_API_URL}/repos/{self.repo}/contents/{self.filename}"
        data = {
            "message": f"Update ledger {datetime.datetime.now()}",
            "content": content_bytes
        }
        if sha:
            data["sha"] = sha
        response = requests.put(url, headers=headers, data=json.dumps(data))
        return response.status_code in [200, 201]

# --- 账单解析与去重类 ---
class BillParser:
    @staticmethod
    def identify_and_parse(file):
        """智能识别文件类型并解析"""
        filename = file.name.lower()
        
        if filename.endswith('.csv'):
            return BillParser._parse_csv(file)
        elif filename.endswith(('.xls', '.xlsx')):
            return BillParser._parse_excel(file)
        elif filename.endswith('.pdf'):
            return BillParser._parse_pdf(file)
        else:
            return None, "不支持的文件格式，请上传 CSV, Excel 或 PDF"

    @staticmethod
    def _parse_csv(file):
        """解析 CSV (适配微信/支付宝)"""
        try:
            content = file.getvalue().decode('utf-8')
        except UnicodeDecodeError:
            file.seek(0)
            content = file.getvalue().decode('gbk', errors='ignore')

        # --- 策略：先判断是哪种账单，再精准定位表头 ---
        
        # 1. 微信特征
        if "微信支付账单明细" in content or "商户单号" in content:
            return BillParser._parse_wechat_content(content)
        
        # 2. 支付宝特征
        # 支付宝通常包含 "支付宝交易记录明细" 或者 列名包含 "商品说明" 和 "对方账号"
        elif "支付宝" in content or "Partner Transaction ID" in content:
            return BillParser._parse_alipay_content(content)
             
        # 默认尝试支付宝解析（容错）
        return BillParser._parse_alipay_content(content)

    @staticmethod
    def _parse_excel(file):
        """解析 Excel (招商银行等)"""
        try:
            df = pd.read_excel(file)
        except Exception as e:
            return None, f"Excel 读取失败: {e}"

        cols = [str(c) for c in df.columns]
        col_str = " ".join(cols)
        
        if "交易日期" in col_str and ("支出" in col_str or "交易金额" in col_str):
            return BillParser._parse_cmb_dataframe(df)
        
        return None, "未识别的 Excel 账单格式。"

    @staticmethod
    def _parse_pdf(file):
        """解析 PDF (招商银行)"""
        try:
            results = []
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    # 提取表格
                    table = page.extract_table()
                    if not table:
                        continue
                    
                    # 寻找表头行 (招行PDF通常有 '记账日期' 或 'Date')
                    header_idx = -1
                    for i, row in enumerate(table):
                        # 清洗 row 中的 None
                        row_text = [str(cell).replace('\n', '') for cell in row if cell]
                        row_str = "".join(row_text)
                        if "记账日期" in row_str or "Date" in row_str and "Currency" in row_str:
                            header_idx = i
                            break
                    
                    if header_idx == -1:
                        continue # 没找到表头，跳过此页

                    # 确定列索引 (基于招行标准PDF格式)
                    # 通常: 记账日期(0), 货币(1), 交易金额(2), 联机余额(3), 交易摘要(4), 对手信息(5)
                    # 注意：有时候可能有额外空列，需要动态匹配
                    headers = [str(h).replace('\n', '').strip() for h in table[header_idx] if h]
                    
                    # 开始解析数据
                    for row in table[header_idx+1:]:
                        # 过滤无效行 (例如下一页的表头或者是空的)
                        if not row or len(row) < 3: continue
                        
                        # 简单映射：假设前几列固定
                        # 清洗换行符
                        clean_row = [str(cell).strip() if cell else "" for cell in row]
                        
                        # 日期列 (通常第1列)
                        date_str = clean_row[0].replace('\n', '')
                        if not re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                            continue # 不是日期，跳过

                        # 金额列 (通常第3列)
                        amt_str = clean_row[2].replace(',', '').replace('\n', '')
                        try:
                            amt = float(amt_str)
                        except:
                            continue

                        final_type = "支出" if amt < 0 else "收入"
                        final_amt = abs(amt)

                        # 备注信息 (摘要 + 对手信息)
                        # 摘要通常第5列，对手信息第6列 (索引4, 5)
                        memo = ""
                        if len(clean_row) > 4:
                            memo += clean_row[4].replace('\n', ' ')
                        if len(clean_row) > 5:
                            memo += " " + clean_row[5].replace('\n', ' ')

                        results.append({
                            "日期": date_str,
                            "类型": final_type,
                            "金额": final_amt,
                            "备注": memo.strip(),
                            "分类": "招行PDF"
                        })
            
            if not results:
                return None, "PDF 解析成功但未提取到有效数据，请确认是招商银行流水。"
                
            return pd.DataFrame(results), None

        except Exception as e:
            return None, f"PDF 解析异常: {str(e)}"

    @staticmethod
    def _parse_wechat_content(content):
        # 微信逻辑优化：寻找 "交易时间" 所在行作为 Header
        lines = content.split('\n')
        start_row = 0
        found = False
        for i, line in enumerate(lines):
            # 微信表头特征：包含 '交易时间' 且包含 '当前状态'
            if "交易时间" in line and "当前状态" in line:
                start_row = i
                found = True
                break
        
        if not found:
            return None, "未找到微信账单表头"

        try:
            df = pd.read_csv(StringIO(content), header=start_row)
        except:
            return None, "微信CSV结构错误"

        df.columns = [c.strip() for c in df.columns]
        
        # 筛选支付成功的
        if '当前状态' in df.columns:
            df = df[df['当前状态'] == '支付成功']
        
        results = []
        for _, row in df.iterrows():
            row_type = row.get('收/支', '')
            if row_type == "/" or row_type == "不计收支": continue
            
            final_type = "支出" if row_type == "支出" else "收入"
            # 处理金额：去 ¥ 符号
            amt_str = str(row.get('金额(元)', 0)).replace('¥', '').replace(',', '')
            try:
                amt = float(amt_str)
            except:
                continue
            
            # 日期处理
            try:
                d_str = pd.to_datetime(row['交易时间']).strftime('%Y-%m-%d')
            except:
                continue

            # 组合备注：商品 + 交易对方
            item = str(row.get('商品', '')).strip()
            partner = str(row.get('交易对方', '')).strip()
            memo = f"{partner} - {item}" if partner else item

            results.append({
                "日期": d_str,
                "类型": final_type,
                "金额": amt,
                "备注": memo.strip(),
                "分类": "微信导入"
            })
        return pd.DataFrame(results), None

    @staticmethod
    def _parse_alipay_content(content):
        # 支付宝逻辑优化
        lines = content.split('\n')
        start_row = 0
        found = False
        for i, line in enumerate(lines):
            # 支付宝表头特征：包含 '交易时间' 且包含 '交易分类' (用户提供的样本特征)
            # 或者包含 '交易时间' 和 '商品说明'
            if "交易时间" in line and ("交易分类" in line or "商品说明" in line):
                start_row = i
                found = True
                break
        
        if not found:
            # 尝试暴力回退查找
            # 有时候分隔线在表头上面
            for i, line in enumerate(lines):
                if "----------------" in line:
                    start_row = i + 1
                    found = True
                    break
        
        if not found:
             return None, "未找到支付宝账单表头"

        try:
            df = pd.read_csv(StringIO(content), header=start_row)
        except:
            return None, "支付宝CSV结构错误"

        df.columns = [c.strip() for c in df.columns]
        
        # 状态过滤
        if '交易状态' in df.columns:
            df = df[df['交易状态'].isin(['交易成功', '支付成功', '已支出', '资金转移'])]

        results = []
        for _, row in df.iterrows():
            # 过滤空金额
            if pd.isna(row.get('金额')): continue
            
            row_type = str(row.get('收/支', '')).strip()
            # 用户样本显示有 "不计收支"，通常我们不记这笔（因为可能是理财/转账），或者记为支出？
            # 按照惯例，"不计收支" 往往是信用卡还款或理财，为了不重记，通常忽略，除非用户强行要
            # 这里保持忽略逻辑
            if row_type == "不计收支" or row_type == "": continue
            
            final_type = "支出" if row_type == "支出" else "收入"
            try:
                amt = float(str(row['金额']))
            except:
                continue
            
            try:
                d_str = pd.to_datetime(row['交易时间']).strftime('%Y-%m-%d')
            except:
                continue
            
            partner = str(row.get('交易对方', '')).strip()
            item_name = str(row.get('商品说明', '')).strip()
            cat = str(row.get('交易分类', '支付宝导入')).strip()

            results.append({
                "日期": d_str,
                "类型": final_type,
                "金额": amt,
                "备注": f"{partner} {item_name}".strip(),
                "分类": cat
            })
        return pd.DataFrame(results), None

    @staticmethod
    def _parse_cmb_dataframe(df):
        """招行 Excel DataFrame 解析"""
        # 寻找 Header
        header_row_idx = 0
        for i in range(len(df)):
            row_vals = [str(v) for v in df.iloc[i].values]
            if "交易日期" in row_vals or "记账日期" in row_vals:
                header_row_idx = i
                break
        
        df.columns = df.iloc[header_row_idx]
        df = df.iloc[header_row_idx+1:]
        df.columns = [str(c).strip() for c in df.columns]
        
        results = []
        for _, row in df.iterrows():
            date_val = row.get('交易日期') or row.get('记账日期')
            if pd.isna(date_val): continue
            
            try:
                d_str = pd.to_datetime(str(date_val)).strftime('%Y-%m-%d')
            except:
                continue

            # 金额处理
            expense = row.get('支出', 0)
            income = row.get('收入', 0)
            trans_amt = row.get('交易金额', 0)

            final_amt = 0.0
            final_type = "支出"
            
            if trans_amt != 0 and not pd.isna(trans_amt):
                # 招行可能是 "-22.00" 字符串
                try:
                    t_val = float(str(trans_amt).replace(',', ''))
                    final_amt = abs(t_val)
                    final_type = "支出" if t_val < 0 else "收入"
                except:
                    pass
            elif expense and float(str(expense).replace(',', '')) > 0:
                final_amt = float(str(expense).replace(',', ''))
                final_type = "支出"
            elif income and float(str(income).replace(',', '')) > 0:
                final_amt = float(str(income).replace(',', ''))
                final_type = "收入"
            
            if final_amt == 0: continue

            memo = str(row.get('交易备注') or row.get('交易摘要') or "") + " " + str(row.get('对手信息') or "")
            
            results.append({
                "日期": d_str,
                "类型": final_type,
                "金额": final_amt,
                "备注": memo.strip(),
                "分类": "招行导入"
            })
            
        return pd.DataFrame(results), None

    @staticmethod
    def merge_and_deduplicate(old_df, new_df):
        """
        合并并去重
        策略：如果 Date + Amount + Type 相同，视为重复。
        强化：招行账单如果备注包含 '支付宝'/'微信'/'财付通' 且金额匹配，视为重复（即使 Old 数据里没有备注）。
        """
        if new_df is None or new_df.empty:
            return old_df, 0, 0

        added_rows = []
        skipped_count = 0
        
        # 建立索引：(日期, 金额) -> 存在的记录列表
        # 使用 set 存储 key 加速判断
        # Key 格式: "2023-01-01_100.50_支出"
        existing_keys = set()
        for _, row in old_df.iterrows():
            try:
                amt = float(row['金额'])
                key = f"{row['日期']}_{amt:.2f}_{row['类型']}"
                existing_keys.add(key)
            except:
                continue

        for _, row in new_df.iterrows():
            try:
                amt = float(row['金额'])
                key = f"{row['日期']}_{amt:.2f}_{row['类型']}"
            except:
                continue
            
            is_duplicate = False
            
            # 1. 严格全匹配检查
            if key in existing_keys:
                is_duplicate = True
                
            # 2. 招商银行特殊去重逻辑 (重叠账单)
            # 如果这笔是招行的，且金额/日期已经在账本里了（大概率是支付宝/微信记过了），且招行备注里明确写了它是第三方支付
            memo = str(row['备注'])
            is_cmb = "招行" in str(row.get('分类', ''))
            is_third_party_payment = any(k in memo for k in ["支付宝", "微信", "财付通", "Tenpay", "Alipay", "美团", "京东", "银联快捷"])
            
            if is_duplicate:
                skipped_count += 1
                continue
            
            # 如果不是严格重复，但属于 [招行] + [第三方支付关键词] + [账本里已有同天同金额记录]
            # 这种情况也要跳过，防止双重记账
            # 注意：这里的逻辑假设“同天同金额”就是同一笔交易，对于小额高频交易（如一天买两次3块钱的水）可能会误杀，
            # 但对于整理“御三家”流水来说，误杀概率低于重复记账的烦恼。
            if is_cmb and is_third_party_payment and key in existing_keys:
                 skipped_count += 1
                 continue

            added_rows.append(row)
            existing_keys.add(key) # 防止本批次内自我重复

        if not added_rows:
            return old_df, 0, skipped_count
            
        return pd.concat([old_df, pd.DataFrame(added_rows)], ignore_index=True), len(added_rows), skipped_count

# --- AI 处理函数 ---
def process_bill_image(image_file, api_key):
    if not api_key:
        return None, "未配置 API Key"

    image_bytes = image_file.getvalue()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = """
    请识别这张账单图片。提取以下字段并以JSON格式返回：
    1. date (格式YYYY-MM-DD)
    2. amount (数字类型，不要带货币符号)
    3. merchant (商户名或交易说明)
    4. category (从以下选择最接近的: 餐饮, 交通, 购物, 居住, 娱乐, 工资, 其他)
    5. type (支出 或 收入)
    
    直接返回JSON，不需要 ```json 标记。
    """

    payload = {
        "model": VISION_MODEL_NAME, 
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 512
    }

    try:
        response = requests.post(
            "[https://api.siliconflow.cn/v1/chat/completions](https://api.siliconflow.cn/v1/chat/completions)",
            headers=headers,
            json=payload,
            timeout=45
        )
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            clean_content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_content), None
        else:
            return None, f"API Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, f"请求异常: {str(e)}"

# --- 主程序 ---
def main():
    # 1. 配置加载
    st.sidebar.title("⚙️ 个人财务设置")
    
    sf_api_key = st.secrets.get("SILICONFLOW_API_KEY", "")
    github_token = st.secrets.get("GITHUB_TOKEN", "")
    github_repo = st.secrets.get("GITHUB_REPO", "")

    dm = DataManager(github_token, github_repo)
    
    if dm.use_github:
        st.sidebar.success(f"☁️ 数据存储: GitHub ({github_repo})")
    else:
        st.sidebar.warning("📂 数据存储: 本地模式")

    payday = st.sidebar.number_input("每月发薪日", 1, 31, 10)
    current_cash = st.sidebar.number_input("当前现金/余额", value=3000.0)

    # 2. 加载数据
    if 'ledger_data' not in st.session_state:
        df, sha = dm.load_data()
        st.session_state.ledger_data = df
        st.session_state.github_sha = sha

    # 3. 财务概览
    st.title("💰 极简账本")
    
    today = date.today()
    if today.day >= payday:
        next_pay_date = date(today.year + (1 if today.month == 12 else 0), 1 if today.month == 12 else today.month + 1, payday)
    else:
        next_pay_date = date(today.year, today.month, payday)
    
    days_left = (next_pay_date - today).days
    
    col1, col2, col3 = st.columns(3)
    col1.metric("当前余额", f"¥{current_cash:,.2f}")
    col2.metric("距离发工资", f"{days_left} 天")
    
    if days_left > 0:
        daily_budget = current_cash / days_left
        gap = daily_budget - DEFAULT_TARGET_SPEND
        col3.metric("每日可用", f"¥{daily_budget:.1f}", 
                    f"{gap:+.1f} (vs ¥{DEFAULT_TARGET_SPEND})",
                    delta_color="normal" if gap >= 0 else "inverse")
    else:
        col3.metric("每日可用", "N/A", "今日发薪！")

    st.divider()

    # 4. 记账功能区 - 统一入口
    tab_auto, tab_manual = st.tabs(["📤 智能导入 (多文件/图片)", "✍️ 手动记账"])

    with tab_auto:
        st.markdown("""
        <small>支持格式：
        1. **图片** (jpg/png) -> AI 自动识别
        2. **文件** (csv/xlsx/xls/pdf) -> 批量导入微信/支付宝/招行账单 (自动合并去重)
        </small>
        """, unsafe_allow_html=True)
        
        # 允许上传多个文件
        uploaded_files = st.file_uploader(
            "点击上传 (支持多选)", 
            type=['png', 'jpg', 'jpeg', 'csv', 'xlsx', 'xls', 'pdf'], 
            key="unified_upload",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # 文件分类
            img_files = [f for f in uploaded_files if f.name.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']]
            data_files = [f for f in uploaded_files if f.name.split('.')[-1].lower() in ['csv', 'xlsx', 'xls', 'pdf']]

            col_a, col_b = st.columns(2)
            
            # --- 批量处理数据文件 ---
            if data_files:
                with col_a:
                    st.info(f"检测到 {len(data_files)} 个数据文件")
                    if st.button(f"批量解析导入", key="btn_import_batch"):
                        total_added = 0
                        total_skipped = 0
                        
                        with st.spinner("正在批量解析..."):
                            batch_df = pd.DataFrame()
                            
                            for f in data_files:
                                df_new, err = BillParser.identify_and_parse(f)
                                if err:
                                    st.error(f"文件 {f.name} 解析失败: {err}")
                                elif df_new is not None and not df_new.empty:
                                    batch_df = pd.concat([batch_df, df_new], ignore_index=True)
                            
                            if not batch_df.empty:
                                merged_df, added_count, skipped_count = BillParser.merge_and_deduplicate(
                                    st.session_state.ledger_data, batch_df
                                )
                                total_added += added_count
                                total_skipped += skipped_count
                                
                                if total_added > 0:
                                    if dm.save_data(merged_df, st.session_state.get('github_sha')):
                                        st.session_state.ledger_data = merged_df
                                        st.session_state.github_sha = dm.load_data()[1]
                                        st.success(f"🎉 批量导入完成！新增 {total_added} 条记录。")
                                        if total_skipped > 0:
                                            st.info(f"🛡️ 自动跳过了 {total_skipped} 条重复或重合记录")
                                        st.rerun()
                                    else:
                                        st.error("保存失败")
                                else:
                                    st.warning(f"所有记录均已存在 (跳过 {total_skipped} 条)。")
                            else:
                                st.warning("没有解析出有效数据。")

            # --- 批量/单张 图片处理 ---
            if img_files:
                with col_b:
                    st.info(f"检测到 {len(img_files)} 张图片")
                    if 'ocr_queue' not in st.session_state:
                        st.session_state.ocr_queue = []
                        
                    if st.button(f"开始 AI 识别 ({len(img_files)}张)", key="btn_ocr_batch"):
                        if not sf_api_key:
                            st.error("请配置 SILICONFLOW_API_KEY")
                        else:
                            with st.spinner("AI 正在逐张读取..."):
                                for img_f in img_files:
                                    data, err = process_bill_image(img_f, sf_api_key)
                                    if not err and data:
                                        data['_filename'] = img_f.name
                                        st.session_state.ocr_queue.append(data)
                                    else:
                                        st.error(f"{img_f.name} 识别失败: {err}")
                            st.rerun()

        # --- OCR 结果确认队列 ---
        if 'ocr_queue' in st.session_state and len(st.session_state.ocr_queue) > 0:
            st.divider()
            st.subheader(f"🔍 待确认 OCR 结果 (剩余 {len(st.session_state.ocr_queue)} 个)")
            
            current_ocr = st.session_state.ocr_queue[0]
            
            with st.container(border=True):
                st.caption(f"来源文件: {current_ocr.get('_filename', 'Unknown')}")
                with st.form("ocr_confirm_queue"):
                    c1, c2 = st.columns(2)
                    o_date = c1.date_input("日期", pd.to_datetime(current_ocr.get('date', str(date.today()))))
                    o_type = c2.selectbox("类型", ["支出", "收入"], index=1 if current_ocr.get('type') == '收入' else 0)
                    o_amt = c1.number_input("金额", float(current_ocr.get('amount', 0)))
                    o_cat = c2.text_input("分类", current_ocr.get('category', '餐饮'))
                    o_desc = st.text_input("备注", current_ocr.get('merchant', ''))
                    
                    col_submit, col_skip = st.columns([1, 1])
                    if col_submit.form_submit_button("✅ 确认添加"):
                        new_row = {"日期": str(o_date), "类型": o_type, "金额": o_amt, "备注": o_desc, "分类": o_cat}
                        st.session_state.ledger_data = pd.concat([st.session_state.ledger_data, pd.DataFrame([new_row])], ignore_index=True)
                        dm.save_data(st.session_state.ledger_data, st.session_state.get('github_sha'))
                        st.session_state.github_sha = dm.load_data()[1]
                        st.session_state.ocr_queue.pop(0)
                        st.rerun()
                        
                    if col_skip.form_submit_button("🗑️ 跳过此条"):
                        st.session_state.ocr_queue.pop(0)
                        st.rerun()

    # --- Manual Tab ---
    with tab_manual:
        with st.form("manual_form"):
            c_m1, c_m2 = st.columns(2)
            m_date = c_m1.date_input("日期", date.today())
            m_type = c_m2.selectbox("类型", ["支出", "收入"])
            m_amt = c_m1.number_input("金额", step=1.0)
            m_cat = c_m2.selectbox("分类", ["餐饮", "交通", "购物", "居住", "娱乐", "工资", "其他"])
            m_desc = st.text_input("备注")
            
            if st.form_submit_button("💾 保存记录"):
                new_row = {"日期": str(m_date), "类型": m_type, "金额": m_amt, "备注": m_desc, "分类": m_cat}
                st.session_state.ledger_data = pd.concat([st.session_state.ledger_data, pd.DataFrame([new_row])], ignore_index=True)
                dm.save_data(st.session_state.ledger_data, st.session_state.get('github_sha'))
                st.session_state.github_sha = dm.load_data()[1]
                st.rerun()

    st.divider()

    # 5. 历史账单 & 可视化
    if not st.session_state.ledger_data.empty:
        st.subheader("📊 历史账单")
        edited_df = st.data_editor(
            st.session_state.ledger_data,
            num_rows="dynamic",
            use_container_width=True,
            key="history_editor"
        )
        if st.button("🔄 同步表格修改"):
            if dm.save_data(edited_df, st.session_state.get('github_sha')):
                st.session_state.ledger_data = edited_df
                st.session_state.github_sha = dm.load_data()[1]
                st.success("同步成功")
                st.rerun()
        
        st.subheader("📈 消费透视")
        chart_df = st.session_state.ledger_data.copy()
        chart_df['金额'] = pd.to_numeric(chart_df['金额'], errors='coerce').fillna(0)
        chart_df['日期'] = pd.to_datetime(chart_df['日期']).dt.date
        expense_df = chart_df[chart_df['类型'] == '支出']
        
        if not expense_df.empty:
            t1, t2 = st.tabs(["📊 分类占比", "📉 每日趋势"])
            with t1:
                st.bar_chart(expense_df.groupby('分类')['金额'].sum().sort_values(ascending=False), color="#FF4B4B")
            with t2:
                st.line_chart(expense_df.groupby('日期')['金额'].sum())
    else:
        st.info("暂无数据")

    # 6. AI 分析
    with st.expander("🤖 AI 财务分析"):
        if st.button("分析我的开销"):
            if sf_api_key and not st.session_state.ledger_data.empty:
                with st.spinner("AI 正在思考..."):
                    summary = st.session_state.ledger_data.to_string()
                    payload = {
                        "model": TEXT_MODEL_NAME, 
                        "messages": [{"role": "user", "content": f"分析这份账单，指出问题：\n{summary}"}]
                    }
                    try:
                        r = requests.post("[https://api.siliconflow.cn/v1/chat/completions](https://api.siliconflow.cn/v1/chat/completions)", 
                                        headers={"Authorization": f"Bearer {sf_api_key}"}, json=payload)
                        st.markdown(r.json()['choices'][0]['message']['content'])
                    except Exception as e:
                        st.error(f"AI 服务异常: {e}")

if __name__ == "__main__":
    main()
