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
import math

# --- 页面配置 ---
st.set_page_config(page_title="AI 智能账本 Pro (大文件版)", page_icon="💰", layout="wide")

# --- 常量配置 ---
DEFAULT_TARGET_SPEND = 60.0  # 每日体面支出标准
GITHUB_API_URL = "https://api.github.com"
VISION_MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"  # 假设使用的视觉模型
TEXT_MODEL_NAME = "deepseek-ai/DeepSeek-V3" # 文本模型
CHUNK_SIZE = 8000  # 智能分片阈值 (字符数)，配合 LLM 的 Context Window

# --- 核心工具：OpenAI Client ---
def get_llm_client(api_key):
    # 请根据实际情况修改 base_url，这里默认使用 SiliconFlow 或类似的兼容接口
    return OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

# --- 工具函数：增强版 JSON 提取与截断修复 ---
def extract_json_from_text(text):
    """
    超强容错 JSON 提取器：
    1. 提取 markdown 代码块
    2. 处理 JSON 截断（缺少 ] 的情况）
    3. 清理注释
    返回: (data_list, error_msg)
    """
    if not text: 
        return None, "空响应"
    
    # 保存原始文本用于调试
    original_preview = text[:200].replace('\n', '\\n')
    
    # 1. 预处理：尝试提取 Markdown 代码块
    try:
        text = text.strip()
        code_block_pattern = r"match_code = re.search(code_block_pattern, text, re.DOTALL)
        if match_code:
            text = match_code.group(1).strip()
        else:
            # 兜底：移除可能的 markdown 标记
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            text = text.strip()
            
        # 2. 快速判断空数组
        if text == '[]':
            return [], None

        # 3. 核心修复：处理截断的 JSON
        # 如果结尾不是 ]，尝试寻找最后一个闭合的大括号 } 并补全 ]
        if not text.endswith(']'):
            last_brace_index = text.rfind('}')
            if last_brace_index != -1:
                # 截取到最后一个完整对象，并补全数组结束符
                text = text[:last_brace_index+1] + ']'
            else:
                # 连一个完整对象都没有
                return None, "未找到有效的JSON对象结尾"

        # 4. 尝试定位数组边界 (处理 AI 回复中包含前后文的情况)
        match_array = re.search(r'\[.*\]', text, re.DOTALL)
        if match_array:
            text_to_parse = match_array.group()
        else:
            text_to_parse = text
            
        # 5. 清理常见的 JS 注释 (// 或 /* */) 防止 json.loads 失败
        text_to_parse = re.sub(r'//.*?\n', '\n', text_to_parse)
        text_to_parse = re.sub(r'/\*.*?\*/', '', text_to_parse, flags=re.DOTALL)
        
        # 6. 正式解析
        result = json.loads(text_to_parse)
        
        if isinstance(result, (list, dict)):
            return result if isinstance(result, list) else [result], None
            
    except json.JSONDecodeError as e:
        return None, f"JSON解析失败 (位置 {e.pos}): {original_preview}...", None
    except Exception as e:
        return None, f"未知解析错误: {str(e)}", None
    
    return None, f"无法提取有效JSON数据", None

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
        expected_cols = ["日期", "类型", "金额", "备注", "分类"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ""
        df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0.0)
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        df['日期'] = df['日期'].fillna(pd.Timestamp(date.today()))
        df['日期'] = df['日期'].dt.date
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
                return self._create_empty_df(), content['sha']
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
                    st.warning(f"⚠️ SHA冲突 ({resp.status_code})，尝试自动修复...")
                self._fetch_github_content.clear()
                latest, _ = self._fetch_github_content()
                if latest and 'sha' in latest:
                    data["sha"] = latest['sha']
                    retry_resp = do_put(data)
                    if retry_resp.status_code in [200, 201]:
                        self._fetch_github_content.clear()
                        return True, retry_resp.json()['content']['sha']
                return False, None
            else:
                return False, None
        except Exception:
            return False, None

    @staticmethod
    def _create_empty_df():
        return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"])

# --- AI 解析核心 (含智能分片) ---
class BillParser:
    
    @staticmethod
    def _split_text_safe(text, chunk_size=CHUNK_SIZE):
        """
        智能分片：按行切割，确保不打断数据行。
        """
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_len = 0
        
        for line in lines:
            line_len = len(line) + 1 # +1 for newline
            if current_len + line_len > chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            
            current_chunk.append(line)
            current_len += line_len
            
        if current_chunk:
            chunks.append("\n".join(current_chunk))
            
        return chunks

    @staticmethod
    def _process_single_chunk(chunk_text, chunk_index, total_chunks, source_type, api_key):
        """处理单个分片"""
        client = get_llm_client(api_key)
        # 针对分片优化的 Prompt
        prompt = f"""
        你是一个严谨的财务数据提取专家。
        任务：从以下文本片段中提取交易记录。
        **注意：这是完整文件的第 {chunk_index + 1}/{total_chunks} 个片段，数据可能在开头或结尾被截断。请尽可能提取完整的记录。**
        
        输入文本类型：{source_type}
        当前年份参考：{datetime.datetime.now().year}
        
        **强制要求**：
        1. 必须返回**纯JSON数组**。
        2. 格式：[{{"date":"YYYY-MM-DD","type":"支出/收入","amount":数字,"merchant":"商户/备注","category":"分类"}}]
        3. 如果片段内无有效完整数据，返回 []
        4. 不要包含任何Markdown标记或解释文字。

        文本内容：
        {chunk_text}
        """
        
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=TEXT_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096, # 预留足够的 Output Token
                temperature=0.0
            )
            raw_json = resp.choices[0].message.content
            data, err = extract_json_from_text(raw_json)
            
            return {
                "chunk_index": chunk_index,
                "data": data,
                "raw_response": raw_json, # 用于调试
                "error": err,
                "time": time.time() - t0
            }
        except Exception as e:
            return {
                "chunk_index": chunk_index,
                "data": None,
                "raw_response": str(e),
                "error": str(e),
                "time": time.time() - t0
            }

    @staticmethod
    def identify_and_parse(filename, file_bytes, api_key):
        """智能入口：根据文件大小决定是否分片并发"""
        t_start = time.time()
        debug_log = {"file": filename, "steps": [], "chunks_info": []}
        
        try:
            # 1. 读取内容
            content_text = ""
            source_type = "未知"
            file_stream = BytesIO(file_bytes)
            
            if filename.endswith('.csv'):
                source_type = "CSV"
                try: content_text = file_bytes.decode('utf-8')
                except: content_text = file_bytes.decode('gbk', errors='ignore')
            elif filename.endswith(('.xls', '.xlsx')):
                source_type = "Excel"
                xls = pd.read_excel(file_stream, sheet_name=None)
                parts = []
                for sname, sdf in xls.items():
                    parts.append(f"Sheet: {sname}\n{sdf.to_csv(index=False)}")
                content_text = "\n".join(parts)
            elif filename.endswith('.pdf'):
                source_type = "PDF"
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    content_text = "\n".join([p.get_text() for p in doc])
            
            total_chars = len(content_text)
            debug_log["steps"].append(f"读取完成，总字符数: {total_chars}")
            
            if not content_text.strip():
                return None, "内容为空", debug_log

            # 2. 智能分片策略
            chunks = BillParser._split_text_safe(content_text, CHUNK_SIZE)
            total_chunks = len(chunks)
            debug_log["steps"].append(f"智能分片: 共 {total_chunks} 个片段")

            all_data = []
            
            # 3. 并发处理分片
            # 限制并发数防止 API Rate Limit
            max_workers = min(5, total_chunks) 
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(BillParser._process_single_chunk, chunk, i, total_chunks, source_type, api_key)
                    for i, chunk in enumerate(chunks)
                ]
                
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    # 记录调试信息
                    chunk_log = {
                        "chunk": res['chunk_index'],
                        "status": "Success" if res['data'] is not None else "Failed",
                        "items_count": len(res['data']) if res['data'] else 0,
                        "error": res['error'],
                        "time": f"{res['time']:.2f}s",
                        "response_preview": res['raw_response'][:100] + "..." if res['raw_response'] else ""
                    }
                    # 如果开启深层调试，保存完整响应
                    if st.session_state.get('debug_mode', False):
                        chunk_log["full_response"] = res['raw_response']
                        
                    debug_log["chunks_info"].append(chunk_log)
                    
                    if res['data']:
                        all_data.extend(res['data'])

            # 4. 合并与清洗
            if not all_data:
                return None, "所有分片均未提取到有效数据", debug_log
                
            df = pd.DataFrame(all_data)
            cols = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
            df = df.rename(columns=cols)
            for c in cols.values():
                if c not in df.columns: df[c] = ""
            
            # 基础清洗
            df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
            df['日期'] = df['日期'].astype(str).apply(lambda x: x.split(' ')[0])
            
            debug_log["total_time"] = time.time() - t_start
            return df, None, debug_log

        except Exception as e:
            return None, str(e), debug_log

    @staticmethod
    def process_image(filename, image_bytes, api_key):
        """图片处理 (保持原逻辑，图片一般不切分)"""
        t_start = time.time()
        debug_log = {"file": filename, "steps": [], "type": "Image"}
        
        try:
            b64_img = base64.b64encode(image_bytes).decode('utf-8')
            client = get_llm_client(api_key)
            
            resp = client.chat.completions.create(
                model=VISION_MODEL_NAME,
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "提取账单明细。返回纯JSON数组：[{date, type, amount, merchant, category}]"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }],
                max_tokens=2048
            )
            raw_json = resp.choices[0].message.content
            if st.session_state.get('debug_mode'):
                debug_log["full_response"] = raw_json
                
            data, parse_err = extract_json_from_text(raw_json)
            
            if parse_err: return None, parse_err, debug_log
            if isinstance(data, dict): data = [data]
            if not data: return None, "无数据", debug_log
            
            df = pd.DataFrame(data)
            cols = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
            df = df.rename(columns=cols)
            for c in cols.values(): 
                if c not in df.columns: df[c] = ""
            
            return df, None, debug_log
        except Exception as e:
            return None, str(e), debug_log

    @staticmethod
    def merge_data(old_df, new_df):
        if new_df is None or new_df.empty: return old_df, 0
        def get_fp(d): return d['日期'].astype(str) + d['金额'].astype(str) + d['备注'].str[:5]
        if old_df.empty: return new_df, len(new_df)
        old_fp = set(get_fp(old_df))
        new_df['_fp'] = get_fp(new_df)
        to_add = new_df[~new_df['_fp'].isin(old_fp)].drop(columns=['_fp'])
        if to_add.empty: return old_df, 0
        merged = pd.concat([old_df, to_add], ignore_index=True)
        merged = DataManager._clean_df_types(merged)
        merged = merged.sort_values('日期', ascending=False).reset_index(drop=True)
        return merged, len(to_add)

# --- 主程序 UI ---
def main():
    if 'debug_mode' not in st.session_state: st.session_state.debug_mode = False
    
    st.sidebar.title("⚙️ 设置")
    st.session_state.debug_mode = st.sidebar.checkbox("🐞 深度调试模式", value=st.session_state.debug_mode, help="显示分片解析详情和AI原始响应")
    
    api_key = st.secrets.get("SILICONFLOW_API_KEY") or st.sidebar.text_input("API Key (SiliconFlow/DeepSeek)", type="password")
    gh_token = st.secrets.get("GITHUB_TOKEN")
    gh_repo = st.secrets.get("GITHUB_REPO")
    
    dm = DataManager(gh_token, gh_repo)
    
    if dm.use_github:
        st.sidebar.success(f"已连接: {dm.repo}")
        if st.sidebar.button("☁️ 强制拉取"):
            with st.spinner("同步中..."):
                df, sha = dm.load_data(force_refresh=True)
                st.session_state.ledger_data = df
                st.session_state.github_sha = sha
                st.rerun()
    else:
        st.sidebar.warning("本地模式")

    payday = st.sidebar.number_input("发薪日", 1, 31, 10)
    current_asset = st.sidebar.number_input("当前资产", value=3000.0)

    if 'ledger_data' not in st.session_state:
        df, sha = dm.load_data()
        st.session_state.ledger_data = df
        st.session_state.github_sha = sha

    st.title("💰 AI 智能账本 Pro (Max)")
    
    # 顶部指标
    df = st.session_state.ledger_data.copy()
    today = date.today()
    if not df.empty:
        if '日期' not in df.columns: df['日期'] = []
        df['dt'] = pd.to_datetime(df['日期'], errors='coerce')
        mask = (df['dt'].dt.month == today.month) & (df['dt'].dt.year == today.year) & (df['类型']=='支出')
        month_spend = df.loc[mask, '金额'].sum()
    else:
        month_spend = 0.0

    target_date = date(today.year + (1 if today.month==12 and today.day>=payday else 0), 
                      today.month if today.day < payday else (today.month % 12) + 1, payday)
    days_left = (target_date - today).days
    daily_budget = current_asset / max(1, days_left)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("资产余额", f"¥{current_asset:,.2f}")
    c2.metric("本月已支", f"¥{month_spend:,.2f}")
    c3.metric("距发薪日", f"{days_left} 天")
    c4.metric("每日可用", f"¥{daily_budget:.0f}", delta=f"{daily_budget - DEFAULT_TARGET_SPEND:.0f}")
    st.divider()

    t_import, t_add, t_history, t_stats = st.tabs(["📥 智能导入", "✍️ 手动记账", "📋 历史明细", "📊 统计报表"])

    with t_import:
        st.info("💡 提示：支持超长 CSV/文本账单。系统会自动分片并发处理，无需手动拆分。")
        files = st.file_uploader("支持 PDF/CSV/Excel/图片", accept_multiple_files=True)
        if files and st.button("🚀 开始智能解析", type="primary"):
            if not api_key:
                st.error("请先配置 API Key")
                st.stop()
            
            # 预读取
            tasks_doc, tasks_img = [], []
            for f in files:
                f.seek(0)
                bytes_data = f.read()
                item = {"name": f.name, "bytes": bytes_data}
                ext = f.name.split('.')[-1].lower()
                if ext in ['png', 'jpg', 'jpeg']: tasks_img.append(item)
                else: tasks_doc.append(item)
            
            new_df = pd.DataFrame()
            all_debug_logs = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_files = len(tasks_doc) + len(tasks_img)
            completed_files = 0
            
            # 这里的 Executor 用于文件级并发，BillParser 内部还有分片级并发
            # 为了避免线程爆炸，这里 max_workers 设小一点
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
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
                        all_debug_logs.append(dbg)
                        
                        if res is not None and not res.empty:
                            new_df = pd.concat([new_df, res], ignore_index=True)
                            st.toast(f"✅ {fname} 解析成功 ({len(res)}条)")
                        else:
                            st.error(f"❌ {fname}: {err}")
                    except Exception as e:
                        st.error(f"❌ {fname} 异常: {e}")
                    
                    completed_files += 1
                    progress_bar.progress(completed_files / total_files)
                    status_text.text(f"处理进度: {completed_files}/{total_files}")

            # 调试看板 (关键更新)
            if st.session_state.debug_mode:
                st.divider()
                st.subheader("🔬 深度调试看板")
                for log in all_debug_logs:
                    with st.expander(f"文件: {log['file']} (耗时 {log.get('total_time', 0):.2f}s)", expanded=False):
                        if 'chunks_info' in log:
                            # 表格化显示分片详情
                            chunk_df = pd.DataFrame(log['chunks_info'])
                            st.markdown("#### 分片处理详情")
                            st.dataframe(chunk_df[['chunk', 'status', 'items_count', 'time', 'error']], use_container_width=True)
                            # 原始 JSON 详情
                            st.markdown("#### 完整调试日志")
                            st.json(log)
                        else:
                            st.json(log)

            # 保存逻辑
            if not new_df.empty:
                merged_df, added = DataManager.merge_data(st.session_state.ledger_data, new_df)
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
                    st.warning("所有记录已存在")

    with t_add:
        with st.form("manual_add"):
            c1, c2, c3 = st.columns(3)
            d = c1.date_input("日期", date.today())
            t = c2.selectbox("类型", ["支出", "收入"])
            a = c3.number_input("金额", min_value=0.01)
            c4, c5 = st.columns([1, 2])
            cat = c4.selectbox("分类", ["餐饮", "交通", "购物", "居住", "娱乐", "医疗", "工资", "其他"])
            rem = c5.text_input("备注")
            if st.form_submit_button("💾 保存", use_container_width=True):
                row = pd.DataFrame([{"日期": str(d), "类型": t, "金额": a, "分类": cat, "备注": rem}])
                merged, _ = DataManager.merge_data(st.session_state.ledger_data, row)
                ok, new_sha = dm.save_data(merged, st.session_state.get('github_sha'))
                if ok:
                    st.session_state.ledger_data = merged
                    st.session_state.github_sha = new_sha
                    st.success("保存成功")
                    st.rerun()

    with t_history:
        if not st.session_state.ledger_data.empty:
            st.session_state.ledger_data = DataManager._clean_df_types(st.session_state.ledger_data)
            edited_df = st.data_editor(
                st.session_state.ledger_data,
                use_container_width=True,
                num_rows="dynamic",
                key="history_editor",
                column_config={
                    "金额": st.column_config.NumberColumn(format="¥%.2f"),
                    "日期": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "类型": st.column_config.SelectboxColumn(options=["支出", "收入"]),
                }
            )
            if st.button("💾 保存变更"):
                if not edited_df.equals(st.session_state.ledger_data):
                    ok, new_sha = dm.save_data(edited_df, st.session_state.get('github_sha'))
                    if ok:
                        st.session_state.ledger_data = edited_df
                        st.session_state.github_sha = new_sha
                        st.success("更新成功")
                else:
                    st.info("无变更")
        else:
            st.info("暂无数据")

    with t_stats:
        if not st.session_state.ledger_data.empty:
            df = st.session_state.ledger_data.copy()
            df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
            df_exp = df[df['类型'] == '支出']
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("分类占比")
                if not df_exp.empty:
                    st.bar_chart(df_exp.groupby("分类")['金额'].sum())
            with c2:
                st.subheader("支出趋势")
                if not df_exp.empty:
                    st.line_chart(df_exp.groupby("日期")['金额'].sum())
            
            st.divider()
            if st.button("🤖 生成分析报告"):
                if not api_key: st.error("需 API Key")
                else:
                    with st.spinner("AI 分析中..."):
                        summary = df_exp.sort_values('日期', ascending=False).head(100).to_csv(index=False)
                        try:
                            client = get_llm_client(api_key)
                            res = client.chat.completions.create(
                                model=TEXT_MODEL_NAME,
                                messages=[
                                    {"role": "system", "content": "犀利理财师。分析消费结构、预警大额支出、给省钱建议。"},
                                    {"role": "user", "content": summary}
                                ]
                            )
                            st.markdown(res.choices[0].message.content)
                        except Exception as e:
                            st.error(str(e))

if __name__ == "__main__":
    main()
