```python
import streamlit as st
import pandas as pd
import datetime
from datetime import date
import requests
import json
import base64
from io import StringIO, BytesIO
import os
import fitz  # PyMuPDF
import re
from openai import OpenAI
import concurrent.futures
import time

# --- 页面配置 ---
st.set_page_config(page_title="AI 智能账本 Pro", page_icon="💰", layout="wide")

# --- 常量配置 ---
DEFAULT_TARGET_SPEND = 60.0
GITHUB_API_URL = "https://api.github.com"
VISION_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
TEXT_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"

# --- 核心工具：OpenAI Client ---
def get_llm_client(api_key):
    return OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

# --- 工具函数：增强版 JSON 提取 ---
def extract_json_from_text(text):
    """增强版JSON提取，返回 (data, error_msg)"""
    if not text:
        return None, "AI返回为空"
  
    # 保存原始文本用于调试
    original_preview = text[:500].replace('\n', '\\n')
  
    try:
        # 1. 移除Markdown代码块
        cleaned = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'```\s*', '', cleaned).strip()
      
        # 2. 明确无数据
        if cleaned in ['[]', '']:
            return [], None
          
        # 3. 尝试解析
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result, None
        elif isinstance(result, dict):
            return [result], None
          
    except Exception as e:
        # 4. 尝试移除注释
        try:
            no_comments = re.sub(r'//.*?\n', '\n', cleaned)
            no_comments = re.sub(r'/\*.*?\*/', '', no_comments, flags=re.DOTALL)
            result = json.loads(no_comments)
            return result if isinstance(result, list) else [result], None
        except:
            pass
  
    return None, f"格式错误: {original_preview[:100]}..."

# --- 数据管理类 ---
class DataManager:
    def __init__(self, github_token=None, repo=None, filename="ledger.csv"):
        self.github_token = github_token
        if repo and repo.startswith("http"):
            self.repo = repo.rstrip("/").split("github.com/")[-1]
        else:
            self.repo = repo
        self.filename = filename
        self.use_github = bool(github_token and self.repo)

    def load_data(self, force_refresh=False):
        if self.use_github:
            if force_refresh:
                self._fetch_github_content.clear()
            df, sha = self._load_from_github()
        else:
            df, sha = self._load_from_local()
      
        df = self._clean_df_types(df)
        return df, sha

    def save_data(self, df, sha=None):
        save_df = df.copy()
        if '日期' in save_df.columns:
            save_df['日期'] = save_df['日期'].astype(str)
          
        if self.use_github:
            success, new_sha = self._save_to_github(save_df, sha)
            return success, new_sha
        else:
            return self._save_to_local(save_df), None

    @staticmethod
    def _clean_df_types(df):
        """清洗数据类型，确保兼容 st.data_editor"""
        expected_cols = ["日期", "类型", "金额", "备注", "分类"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ""
      
        # 金额转换
        df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0.0)
      
        # 日期转换（更健壮）
        if '日期' in df.columns and not df['日期'].empty:
            if not (pd.api.types.is_datetime64_any_dtype(df['日期']) or 
                    (pd.api.types.is_object_dtype(df['日期']) and 
                     len(df['日期']) > 0 and 
                     isinstance(df['日期'].iloc[0], date))):
                df['日期'] = pd.to_datetime(df['日期'].astype(str), errors='coerce').dt.date
            df['日期'] = df['日期'].fillna(date.today())

        # 字符串列处理
        df['类型'] = df['类型'].astype(str).replace('nan', '支出')
        df['分类'] = df['分类'].astype(str).replace('nan', '其他')
        df['备注'] = df['备注'].astype(str).replace('nan', '')

        return df

    def _load_from_local(self):
        if os.path.exists(self.filename):
            try:
                return pd.read_csv(self.filename), None
            except:
                pass
        return self._create_empty_df(), None

    def _save_to_local(self, df):
        df.to_csv(self.filename, index=False)
        return True

    @st.cache_data(ttl=300, show_spinner=False)
    def _fetch_github_content(_self):
        headers = {
            "Authorization": f"token {_self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        url = f"{GITHUB_API_URL}/repos/{_self.repo}/contents/{_self.filename}"
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.json(), None
            elif response.status_code == 404:
                return None, 404
            else:
                return None, response.status_code
        except Exception as e:
            return None, str(e)

    def _load_from_github(self):
        content, error = self._fetch_github_content()
        if content:
            try:
                csv_str = base64.b64decode(content['content']).decode('utf-8')
                df = pd.read_csv(StringIO(csv_str))
                return df, content['sha']
            except:
                return self._create_empty_df(), content.get('sha')
        return self._create_empty_df(), None

    def _save_to_github(self, df, sha):
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        url = f"{GITHUB_API_URL}/repos/{self.repo}/contents/{self.filename}"
      
        csv_str = df.to_csv(index=False)
        content_bytes = base64.b64encode(csv_str.encode('utf-8')).decode('utf-8')
      
        data = {
            "message": f"Update ledger {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "content": content_bytes
        }
        if sha:
            data["sha"] = sha

        def do_put(payload):
            return requests.put(url, headers=headers, data=json.dumps(payload), timeout=30)

        try:
            resp = do_put(data)
          
            if resp.status_code in [200, 201]:
                self._fetch_github_content.clear()
                return True, resp.json()['content']['sha']
          
            elif resp.status_code in [409, 422]:
                if st.session_state.get('debug_mode'):
                    st.warning(f"⚠️ GitHub SHA冲突({resp.status_code})，正在修复...")
              
                self._fetch_github_content.clear()
                latest_content, _ = self._fetch_github_content()
              
                if latest_content and 'sha' in latest_content:
                    data["sha"] = latest_content['sha']
                    retry_resp = do_put(data)
                    if retry_resp.status_code in [200, 201]:
                        self._fetch_github_content.clear()
                        if st.session_state.get('debug_mode'):
                            st.success("✅ 自动修复成功！")
                        return True, retry_resp.json()['content']['sha']
              
                st.error("❌ 自动修复失败")
                return False, None
          
            else:
                st.error(f"GitHub保存失败: {resp.status_code} - {resp.text}")
                return False, None

        except Exception as e:
            st.error(f"网络异常: {e}")
            return False, None

    @staticmethod
    def _create_empty_df():
        return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"])

# --- AI 解析器 ---
class BillParser:
    @staticmethod
    def identify_and_parse(filename, file_bytes, api_key):
        t_start = time.time()
        debug_log = {"file": filename, "steps": []}
      
        try:
            # 1. 读取内容
            t0 = time.time()
            content_text = ""
            source_type = "未知"
          
            file_stream = BytesIO(file_bytes)
          
            if filename.endswith('.csv'):
                source_type = "CSV"
                try:
                    content_text = file_bytes.decode('utf-8')
                except:
                    try:
                        content_text = file_bytes.decode('gbk')
                    except:
                        content_text = file_bytes.decode('latin-1', errors='ignore')
            elif filename.endswith(('.xls', '.xlsx')):
                source_type = "Excel"
                xls = pd.read_excel(file_stream, sheet_name=None)
                parts = []
                for sname, sdf in xls.items():
                    parts.append(f"Sheet: {sname\n{sdf.to_csv(index=False)}")
                content_text = "\n".join(parts)
            elif filename.endswith('.pdf'):
                source_type = "PDF"
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    content_text = "\n".join([p.get_text() for p in doc])
          
            debug_log["steps"].append(f"读取耗时: {time.time()-t0:.4f}s")
            debug_log["text_len"] = len(content_text)
          
            if not content_text.strip():
                return None, "内容为空", debug_log

            # 2. 智能截断（保留最近200行）
            max_lines = 200
            lines = content_text.split('\n')
            if len(lines) > max_lines:
                content_text = '\n'.join(lines[-max_lines:])
                debug_log["steps"].append(f"⚠️ 文本过长，保留最后{max_lines}行")
          
            max_chars = 50000
            if len(content_text) > max_chars:
                content_text = content_text[-max_chars:] + "\n...(truncated)..."
                debug_log["steps"].append(f"⚠️ 进一步截断到{max_chars}字符")

            # 3. AI处理
            t1 = time.time()
            prompt = f"""
你是一个严谨的财务数据提取专家。
任务：从文本中提取交易记录。
原则：宁缺毋假，禁止捏造。

输入文本类型：{source_type}
当前年份参考：{datetime.datetime.now().year}

**强制要求**：
1. 必须返回纯JSON数组，不要任何解释、markdown或注释
2. 格式：[{{"date":"2024-01-01","type":"支出","amount":123.45,"merchant":"商户","category":"餐饮"}}]
3. 无数据时返回：[]

文本内容：
{content_text}
"""
          
            client = get_llm_client(api_key)
            resp = client.chat.completions.create(
                model=TEXT_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.0
            )
            debug_log["steps"].append(f"AI响应耗时: {time.time()-t1:.4f}s")
          
            # 4. 解析结果
            t2 = time.time()
            raw_json = resp.choices[0].message.content
            debug_log["ai_response_preview"] = raw_json[:500]  # 关键调试信息
            data, parse_error = extract_json_from_text(raw_json)
            debug_log["json_parse_error"] = parse_error
            debug_log["steps"].append(f"JSON解析耗时: {time.time()-t2:.4f}s")
            debug_log["total_time"] = time.time() - t_start
          
            if not data: 
                return None, parse_error or "未提取到有效数据", debug_log
              
            df = pd.DataFrame(data)
            cols = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
            df = df.rename(columns=cols)
            for c in cols.values():
                if c not in df.columns: df[c] = ""
          
            # 清洗
            df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
            df['日期'] = df['日期'].astype(str).apply(lambda x: x.split(' ')[0])
          
            return df, None, debug_log

        except Exception as e:
            debug_log["total_time"] = time.time() - t_start
            debug_log["exception"] = str(e)
            return None, str(e), debug_log

    @staticmethod
    def process_image(filename, image_bytes, api_key):
        t_start = time.time()
        debug_log = {"file": filename, "steps": []}
      
        try:
            b64_img = base64.b64encode(image_bytes).decode('utf-8')
            client = get_llm_client(api_key)
          
            t1 = time.time()
            resp = client.chat.completions.create(
                model=VISION_MODEL_NAME,
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "提取账单明细。必须返回纯JSON数组：[{date, type, amount, merchant, category}]，无数据返回[]"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }],
                max_tokens=2048,
                temperature=0.0
            )
            debug_log["steps"].append(f"视觉模型耗时: {time.time()-t1:.4f}s")
          
            raw_json = resp.choices[0].message.content
            debug_log["ai_response_preview"] = raw_json[:500]
            data, parse_error = extract_json_from_text(raw_json)
            debug_log["json_parse_error"] = parse_error
          
            if not data: return None, parse_error or "识别失败", debug_log
          
            df = pd.DataFrame(data)
            cols = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
            df = df.rename(columns=cols)
            for c in cols.values(): 
                if c not in df.columns: df[c] = ""
          
            debug_log["total_time"] = time.time() - t_start
            return df, None, debug_log
          
        except Exception as e:
            debug_log["total_time"] = time.time() - t_start
            debug_log["exception"] = str(e)
            return None, str(e), debug_log

    @staticmethod
    def merge_data(old_df, new_df):
        """合并去重"""
        if new_df is None or new_df.empty: return old_df, 0
      
        def get_fp(d): 
            return d['日期'].astype(str) + d['金额'].astype(str) + d['备注'].str[:5]
          
        if old_df.empty: 
            new_df_clean = DataManager._clean_df_types(new_df)
            return new_df_clean, len(new_df_clean)
          
        old_fp = set(get_fp(old_df))
        new_df_clean = DataManager._clean_df_types(new_df)
        new_df_clean['_fp'] = get_fp(new_df_clean)
      
        to_add = new_df_clean[~new_df_clean['_fp'].isin(old_fp)].drop(columns=['_fp'])
      
        if to_add.empty: return old_df, 0
      
        merged = pd.concat([old_df, to_add], ignore_index=True)
        merged = DataManager._clean_df_types(merged)
        merged = merged.sort_values('日期', ascending=False).reset_index(drop=True)
        return merged, len(to_add)

# --- 主程序 ---
def main():
    if 'debug_mode' not in st.session_state: st.session_state.debug_mode = False
  
    st.sidebar.title("⚙️ 设置")
    st.session_state.debug_mode = st.sidebar.checkbox("🐞 开启深度调试", value=st.session_state.debug_mode)
  
    api_key = st.secrets.get("SILICONFLOW_API_KEY") or st.sidebar.text_input("API Key", type="password")
    gh_token = st.secrets.get("GITHUB_TOKEN")
    gh_repo = st.secrets.get("GITHUB_REPO")
  
    dm = DataManager(gh_token, gh_repo)
  
    if dm.use_github:
        st.sidebar.success(f"已连接: {dm.repo}")
        if st.sidebar.button("☁️ 强制同步云端"):
            with st.spinner("正在拉取最新数据..."):
                df, sha = dm.load_data(force_refresh=True)
                st.session_state.ledger_data = df
                st.session_state.github_sha = sha
                st.success("同步完成！")
                st.rerun()
    else:
        st.sidebar.warning("本地模式 (数据不持久化)")

    payday = st.sidebar.number_input("每月发薪日", 1, 31, 10)
    current_asset = st.sidebar.number_input("当前资产", value=3000.0)

    # 数据加载
    if 'ledger_data' not in st.session_state:
        df, sha = dm.load_data()
        st.session_state.ledger_data = df
        st.session_state.github_sha = sha

    # 顶部概览
    st.title("💰 AI 智能账本 Pro")
  
    today = date.today()
    target_month = today.month if today.day < payday else (today.month % 12) + 1
    target_year = today.year + (1 if (today.month==12 and today.day >= payday) else 0)
    target_date = date(target_year, target_month, payday)
    days_left = (target_date - today).days

    df = st.session_state.ledger_data.copy()
    month_spend = 0.0
    if not df.empty:
        if '日期' not in df.columns:
             df['日期'] = []
        df['dt'] = pd.to_datetime(df['日期'], errors='coerce')
        mask = (df['dt'].dt.month == today.month) & (df['dt'].dt.year == today.year) & (df['类型']=='支出')
        month_spend = df.loc[mask, '金额'].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("资产余额", f"¥{current_asset:,.2f}")
    c2.metric("本月已支", f"¥{month_spend:,.2f}")
    c3.metric("距发薪日", f"{days_left} 天")
  
    daily_budget = current_asset / max(1, days_left)
    c4.metric("每日可用", f"¥{daily_budget:.0f}", 
              delta=f"{daily_budget - DEFAULT_TARGET_SPEND:.0f}", delta_color="normal")

    st.divider()

    # 主要功能区
    t_import, t_add, t_history, t_stats = st.tabs(["📥 智能导入", "✍️ 手动记账", "📋 历史明细", "📊 统计报表"])

    # --- 智能导入 Tab ---
    with t_import:
        files = st.file_uploader("支持 PDF/CSV/Excel/图片", accept_multiple_files=True)
        if files and st.button("🚀 批量开始识别", type="primary"):
            if not api_key:
                st.error("缺少 API Key")
                st.stop()
          
            tasks_doc = []
            tasks_img = []
          
            with st.status("正在预处理文件...") as status:
                for f in files:
                    ext = f.name.split('.')[-1].lower()
                    f.seek(0)
                    bytes_data = f.read()
                  
                    item = {"name": f.name, "bytes": bytes_data}
                    if ext in ['png', 'jpg', 'jpeg']:
                        tasks_img.append(item)
                    else:
                        tasks_doc.append(item)
                status.update(label="文件读取完成，准备提交 AI", state="complete")

            # 并发处理
            new_df = pd.DataFrame()
            debug_logs = []
            progress = st.progress(0)
            total_tasks = len(tasks_doc) + len(tasks_img)
            completed = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                for t in tasks_doc:
                    f = executor.submit(BillParser.identify_and_parse, t['name'], t['bytes'], api_key)
                    futures[f] = t['name']
                for t in tasks_img:
                    f = executor.submit(BillParser.process_image, t['name'], t['bytes'], api_key)
                    futures[f] = t['name']
              
                for future in concurrent.futures.as_completed(futures):
                    fname = futures[future]
                    try:
                        res, err, dbg = future.result()
                        debug_logs.append(dbg)
                      
                        if res is not None and not res.empty:
                            new_df = pd.concat([new_df, res], ignore_index=True)
                            st.toast(f"✅ {fname} 成功")
                        else:
                            st.error(f"❌ {fname}: {err}")
                          
                    except Exception as e:
                        st.error(f"❌ {fname} 异常: {e}")
                  
                    completed += 1
                    progress.progress(completed / total_tasks)

            # 显示调试信息
            if st.session_state.debug_mode:
                with st.expander("🔬 深度调试信息", expanded=True):
                    st.json(debug_logs)

            # 保存
            if not new_df.empty:
                merged_df, added = BillParser.merge_data(st.session_state.ledger_data, new_df)
                if added > 0:
                    with st.spinner("正在保存至云端..."):
                        ok, new_sha = dm.save_data(merged_df, st.session_state.get('github_sha'))
                        if ok:
                            st.session_state.ledger_data = merged_df
                            st.session_state.github_sha = new_sha
                            st.success(f"🎉 成功存入 {added} 条新记录！")
                        else:
                            st.error("保存失败")
                else:
                    st.warning("所有记录已存在，无新增。")

    # --- 手动记账 Tab ---
    with t_add:
        with st.form("manual_add"):
            c1, c2, c3 = st.columns(3)
            d = c1.date_input("日期", date.today())
            t = c2.selectbox("类型", ["支出", "收入"])
            a = c3.number_input("金额", min_value=0.01)
            c4, c5 = st.columns([1, 2])
            cat = c4.selectbox("分类", ["餐饮", "交通", "购物", "居住", "娱乐", "医疗", "工资", "其他"])
            rem = c5.text_input("备注")
          
            if st.form_submit_button("💾 保存"):
                row = pd.DataFrame([{"日期": str(d), "类型": t, "金额": a, "分类": cat, "备注": rem}])
                merged, added = BillParser.merge_data(st.session_state.ledger_data, row)
                ok, new_sha = dm.save_data(merged, st.session_state.get('github_sha'))
                if ok:
                    st.session_state.ledger_data = merged
                    st.session_state.github_sha = new_sha
                    st.success("保存成功")
                    st.rerun()

    # --- 历史明细 Tab ---
    with t_history:
        st.subheader("📋 账单明细 (支持编辑)")
        if st.session_state.ledger_data.empty:
            st.info("暂无数据")
        else:
            # 创建编辑器专用副本并确保类型正确
            df_for_editor = DataManager._clean_df_types(st.session_state.ledger_data.copy())
          
            if st.session_state.debug_mode:
                st.write("数据类型检查:", df_for_editor.dtypes)
          
            # 修复：使用 width='stretch' 替代 use_container_width
            edited_df = st.data_editor(
                df_for_editor,
                width='stretch',
                num_rows="dynamic",
                key="history_editor",
                column_config={
                    "金额": st.column_config.NumberColumn(format="¥%.2f", required=True),
                    "日期": st.column_config.DateColumn(format="YYYY-MM-DD", required=True),
                    "类型": st.column_config.SelectboxColumn(options=["支出", "收入"], required=True),
                    "分类": st.column_config.SelectboxColumn(options=["餐饮", "交通", "购物", "居住", "娱乐", "医疗", "工资", "其他"]),
                    "备注": st.column_config.TextColumn()
                }
            )
          
            # 编辑器返回后需要再次清洗类型
            edited_df_cleaned = DataManager._clean_df_types(edited_df.copy())
          
            if st.button("💾 保存表格变更"):
                if not edited_df_cleaned.equals(df_for_editor):
                    with st.spinner("同步中..."):
                        ok, new_sha = dm.save_data(edited_df_cleaned, st.session_state.get('github_sha'))
                        if ok:
                            st.session_state.ledger_data = edited_df_cleaned
                            st.session_state.github_sha = new_sha
                            st.success("✅ 更新成功")
                            st.rerun()
                else:
                    st.info("数据未变更")

    # --- 统计报表 Tab ---
    with t_stats:
        if st.session_state.ledger_data.empty:
            st.info("暂无数据，请先记账")
        else:
            df = st.session_state.ledger_data.copy()
            df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
            df_exp = df[df['类型'] == '支出']
          
            c_s1, c_s2 = st.columns(2)
          
            with c_s1:
                st.subheader("📊 分类支出占比")
                if not df_exp.empty:
                    chart_data = df_exp.groupby("分类")['金额'].sum().reset_index()
                    st.bar_chart(chart_data, x="分类", y="金额", color="分类")
                else:
                    st.caption("无支出数据")

            with c_s2:
                st.subheader("📉 每日支出趋势")
                if not df_exp.empty:
                    daily_data = df_exp.groupby("日期")['金额'].sum().reset_index()
                    st.line_chart(daily_data, x="日期", y="金额")
                else:
                    st.caption("无支出数据")

            # AI 分析
            st.divider()
            st.subheader("🤖 AI 财务顾问")
            if st.button("生成本月分析报告"):
                if not api_key:
                    st.error("请配置 API Key")
                else:
                    with st.spinner("AI 正在分析..."):
                        summary_csv = df_exp.sort_values('日期', ascending=False).head(100).to_csv(index=False)
                        client = get_llm_client(api_key)
                        try:
                            res = client.chat.completions.create(
                                model=TEXT_MODEL_NAME,
                                messages=[
                                    {"role": "system", "content": "你是一个犀利的理财师。请根据用户最近的支出（CSV格式），给出：1. 消费结构评价 2. 异常大额支出预警 3. 具体的省钱建议。风格幽默犀利。"},
                                    {"role": "user", "content": summary_csv}
                                ],
                                max_tokens=2000
                            )
                            st.markdown(res.choices[0].message.content)
                        except Exception as e:
                            st.error(f"AI分析失败: {e}")

if __name__ == "__main__":
    main()
```
