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
from openai import OpenAI, APITimeoutError
import concurrent.futures
import time

# --- 页面配置 ---
st.set_page_config(page_title="AI 智能账本 Pro", page_icon="💰", layout="wide")

# --- 常量配置 ---
DEFAULT_TARGET_SPEND = 60.0  # 每日体面支出标准
GITHUB_API_URL = "https://api.github.com"
VISION_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct" 
TEXT_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"

# --- 核心工具：OpenAI Client (无缓存，防止线程安全问题) ---
def get_llm_client(api_key):
    return OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

# --- 工具函数：增强版 JSON 提取（含截断修复） ---
def extract_json_from_text(text, auto_repair=False):
    """增强版JSON提取，支持更多异常格式，返回 (data, error_msg)"""
    if not text: 
        return None, "空响应"
    
    # 保存原始文本用于调试
    original_preview = text[:200].replace('\n', '\\n')
    
    try:
        # 1. 移除所有Markdown代码块标记
        text = text.strip()
        # 安全构建正则表达式，避免三个反引号破坏 Markdown 渲染
        # 匹配 ```json ... ``` 或 ``` ... ```
        code_block_pattern = r"``" + r"`(?:json)?(.*?)``" + r"`"
        match_code = re.search(code_block_pattern, text, re.DOTALL)
        
        if match_code:
            text = match_code.group(1).strip()
        else:
            # 简单的移除兜底
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            text = text.strip()
        
        # 2. 如果内容是[]，直接返回
        if text == '[]':
            return [], None

        # 3. 尝试定位数组边界 (处理 AI 回复中包含前后文的情况)
        match_array = re.search(r'\[.*\]', text, re.DOTALL)
        if match_array:
            text_to_parse = match_array.group()
        else:
            text_to_parse = text
            
        # 4. 尝试直接解析
        result = json.loads(text_to_parse)
        if isinstance(result, (list, dict)):
            return result if isinstance(result, list) else [result], None
            
    except Exception as e:
        # 5. 尝试修复常见问题：删除注释
        try:
            # 移除 // 注释
            text_no_comments = re.sub(r'//.*?\n', '\n', text)
            # 移除 /* */ 注释
            text_no_comments = re.sub(r'/\*.*?\*/', '', text_no_comments, flags=re.DOTALL)
            
            # 再次尝试定位数组
            match_array = re.search(r'\[.*\]', text_no_comments, re.DOTALL)
            if match_array:
                text_no_comments = match_array.group()
            
            result = json.loads(text_no_comments)
            return result if isinstance(result, list) else [result], None
        except:
            pass

        # 6. **JSON截断自动修复**
        if auto_repair:
            try:
                # 检测是否为截断的数组
                if text_to_parse.strip().endswith(','):
                    # 可能是数组元素未完整
                    last_bracket = text_to_parse.rfind('{')
                    if last_bracket != -1:
                        # 移除最后一个不完整的对象
                        text_to_parse = text_to_parse[:last_bracket].strip()
                        if text_to_parse.endswith(','):
                            text_to_parse = text_to_parse[:-1]
                        # 关闭数组
                        text_to_parse += ']'
                        result = json.loads(text_to_parse)
                        return result, None
                elif text_to_parse.strip().endswith('}'):
                    # 可能是最后一个对象完整但缺少闭合括号
                    text_to_parse += ']'
                    result = json.loads(text_to_parse)
                    return result, None
                elif not text_to_parse.strip().endswith(']'):
                    # 其他情况，尝试直接补全
                    text_to_parse = text_to_parse.strip() + ']'
                    result = json.loads(text_to_parse)
                    return result, None
            except:
                pass
    
    return None, f"解析失败，非标准JSON格式。预览: {original_preview}..."

# --- 数据管理类 (增强版) ---
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
        """加载数据，支持强制刷新"""
        if self.use_github:
            if force_refresh:
                # 清除 Streamlit 缓存
                self._fetch_github_content.clear()
            df, sha = self._load_from_github()
        else:
            df, sha = self._load_from_local()
        
        # 统一进行类型清洗，防止 data_editor 报错
        df = self._clean_df_types(df)
        return df, sha

    def save_data(self, df, sha=None):
        """保存数据，带自动重试机制"""
        # 保存前确保格式化为字符串，方便 CSV 存储
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
        # 1. 补全列
        expected_cols = ["日期", "类型", "金额", "备注", "分类"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ""
        
        # 2. 强制转换金额为 float (处理空字符串、非数字字符)
        df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0.0)
        
        # 3. 强制转换日期 (修复 StreamlitAPIException)
        # 先转换为 datetime64[ns]，处理无效值为 NaT
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        # 填充 NaT 为今天
        df['日期'] = df['日期'].fillna(pd.Timestamp(date.today()))
        # 最后转换为 python date 对象
        df['日期'] = df['日期'].dt.date

        # 4. 字符串列处理 NaNs
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

    # 使用 st.cache_data 减少 GitHub API 调用
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
        """
        核心保存逻辑，包含 409/422 冲突自动修复
        """
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
            # 第一次尝试
            resp = do_put(data)
            
            # 如果成功
            if resp.status_code in [200, 201]:
                self._fetch_github_content.clear() # 清除读缓存
                return True, resp.json()['content']['sha']
            
            # 如果失败是因为 SHA 不匹配 (409 Conflict 或 422 Unprocessable Entity)
            elif resp.status_code in [409, 422]:
                if st.session_state.get('debug_mode'):
                    st.warning(f"⚠️ GitHub SHA 冲突 ({resp.status_code})，正在尝试自动修复...")
                
                # 1. 强制获取最新 SHA
                self._fetch_github_content.clear()
                latest_content, _ = self._fetch_github_content()
                
                if latest_content and 'sha' in latest_content:
                    # 2. 更新 payload 中的 sha
                    data["sha"] = latest_content['sha']
                    # 3. 重试保存
                    retry_resp = do_put(data)
                    if retry_resp.status_code in [200, 201]:
                        self._fetch_github_content.clear()
                        if st.session_state.get('debug_mode'):
                            st.success("✅ 自动修复成功！")
                        return True, retry_resp.json()['content']['sha']
                    else:
                        st.error(f"❌ 自动修复失败: {retry_resp.status_code} - {retry_resp.text}")
                        return False, None
                else:
                    st.error("❌ 无法获取最新 SHA，保存失败。")
                    return False, None
            else:
                st.error(f"GitHub 保存失败: {resp.status_code} - {resp.text}")
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
        """处理文件内容，支持大文件智能分片并发处理"""
        t_start = time.time()
        debug_log = {"file": filename, "steps": [], "chunks": []}
        
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
                    parts.append(f"Sheet: {sname}\n{sdf.to_csv(index=False)}")
                content_text = "\n".join(parts)
            elif filename.endswith('.pdf'):
                source_type = "PDF"
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    content_text = "\n".join([p.get_text() for p in doc])
            
            debug_log["steps"].append(f"读取耗时: {time.time()-t0:.4f}s")
            debug_log["text_len"] = len(content_text)
            
            if not content_text.strip():
                return None, "内容为空", debug_log

            # 2. 智能分片与并发处理
            t1 = time.time()
            all_results = []
            
            # 分片阈值：10k 字符
            CHUNK_SIZE = 8000  # 每片约 8k 字符，留有余量
            if len(content_text) > 10000:
                debug_log["steps"].append(f"检测到长文本 ({len(content_text)} 字符)，启动智能分片...")
                
                # 按行分割，保持完整性
                lines = content_text.split('\n')
                chunks = []
                current_chunk = []
                current_len = 0
                
                for line in lines:
                    line_len = len(line) + 1  # +1 for \n
                    if current_len + line_len > CHUNK_SIZE and current_chunk:
                        # 当前块已满，保存并开始新块
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = [line]
                        current_len = line_len
                    else:
                        current_chunk.append(line)
                        current_len += line_len
                
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                
                debug_log["chunk_count"] = len(chunks)
                debug_log["steps"].append(f"分片完成: {len(chunks)} 个片段")
                
                # 并发处理每个分片
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:  # max_workers=3 避免 API 限流
                    future_to_chunk = {}
                    for idx, chunk in enumerate(chunks):
                        # 为每个分片添加上下文提示
                        chunk_prompt = f"""
                        你是一个严谨的财务数据提取专家。
                        任务：从以下文本片段中提取交易记录（这是第 {idx+1}/{len(chunks)} 个片段）。
                        原则：宁缺毋假，禁止捏造。
                        
                        输入文本类型：{source_type}
                        当前年份参考：{datetime.datetime.now().year}
                        
                        **强制要求**：必须返回**纯JSON数组**，不要任何解释。
                        格式：[{{"date":"YYYY-MM-DD","type":"支出/收入","amount":数字,"merchant":"商户/备注","category":"分类"}}]
                        如果本片段无数据，返回：[]
                        
                        文本内容：
                        {chunk}
                        """
                        
                        future = executor.submit(
                            BillParser._process_chunk,
                            chunk_prompt,
                            api_key,
                            idx
                        )
                        future_to_chunk[future] = idx
                    
                    # 收集结果
                    for future in concurrent.futures.as_completed(future_to_chunk):
                        chunk_idx = future_to_chunk[future]
                        try:
                            chunk_result, chunk_debug = future.result(timeout=60)
                            debug_log["chunks"].append(chunk_debug)
                            
                            if chunk_result is not None and not chunk_result.empty:
                                all_results.append(chunk_result)
                        except Exception as e:
                            chunk_debug = {
                                "index": chunk_idx,
                                "status": "error",
                                "error": str(e)
                            }
                            debug_log["chunks"].append(chunk_debug)
                            st.warning(f"第 {chunk_idx+1} 个分片处理失败: {e}")
                
            else:
                # 短文本，直接处理
                prompt = f"""
                你是一个严谨的财务数据提取专家。
                任务：从文本中提取交易记录。
                原则：宁缺毋假，禁止捏造。
                
                输入文本类型：{source_type}
                当前年份参考：{datetime.datetime.now().year}
                
                **强制要求**：必须返回**纯JSON数组**，不要任何解释。
                格式：[{{"date":"YYYY-MM-DD","type":"支出/收入","amount":数字,"merchant":"商户/备注","category":"分类"}}]
                如果无数据，返回：[]
                
                文本内容：
                {content_text}
                """
                
                result, chunk_debug = BillParser._process_chunk(prompt, api_key, 0)
                debug_log["chunks"].append(chunk_debug)
                if result is not None:
                    all_results.append(result)
            
            debug_log["steps"].append(f"AI处理总耗时: {time.time()-t1:.4f}s")
            
            # 3. 合并所有分片结果
            if not all_results:
                return None, "所有分片均未提取到有效数据", debug_log
            
            # 合并所有 DataFrame
            combined_df = pd.concat(all_results, ignore_index=True)
            
        else:
            # 短文本路径
            combined_df, err, debug_log = BillParser._process_short_text(content_text, source_type, api_key, debug_log)
            if err:
                return None, err, debug_log

        # 4. 去重和清洗
        combined_df['金额'] = pd.to_numeric(combined_df['金额'], errors='coerce').fillna(0)
        combined_df['日期'] = combined_df['日期'].astype(str).apply(lambda x: x.split(' ')[0])
        
        debug_log["total_records"] = len(combined_df)
        debug_log["total_time"] = time.time() - t_start
        
        return combined_df, None, debug_log

    except Exception as e:
        return None, str(e), debug_log

    @staticmethod
    def _process_short_text(content_text, source_type, api_key, debug_log):
        """处理短文本（兼容旧逻辑）"""
        t1 = time.time()
        prompt = f"""
        你是一个严谨的财务数据提取专家。
        任务：从文本中提取交易记录。
        原则：宁缺毋假，禁止捏造。
        
        输入文本类型：{source_type}
        当前年份参考：{datetime.datetime.now().year}
        
        **强制要求**：必须返回**纯JSON数组**，不要任何解释。
        格式：[{{"date":"YYYY-MM-DD","type":"支出/收入","amount":数字,"merchant":"商户/备注","category":"分类"}}]
        如果无数据，返回：[]
        
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
        
        raw_json = resp.choices[0].message.content
        debug_log["ai_response_full"] = raw_json
        
        data, parse_err = extract_json_from_text(raw_json, auto_repair=True)
        
        if parse_err: 
            return None, f"JSON提取失败: {parse_err}", debug_log
            
        if not data: 
            return None, "未提取到有效数据", debug_log
            
        df = pd.DataFrame(data)
        # 标准化
        cols = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
        df = df.rename(columns=cols)
        for c in cols.values():
            if c not in df.columns: df[c] = ""
        
        # 清洗
        df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
        df['日期'] = df['日期'].astype(str).apply(lambda x: x.split(' ')[0])
        
        debug_log["total_time"] = time.time() - time.time()
        return df, None, debug_log

    @staticmethod
    def _process_chunk(prompt, api_key, chunk_idx):
        """处理单个分片（独立函数便于并发）"""
        chunk_debug = {"index": chunk_idx, "status": "processing"}
        
        client = get_llm_client(api_key)
        resp = client.chat.completions.create(
            model=TEXT_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.0,
            timeout=60  # 增加超时时间
        )
        
        raw_json = resp.choices[0].message.content
        chunk_debug["ai_response_full"] = raw_json[:500]  # 保留前500字符用于调试
        
        # 提取和修复 JSON
        data, parse_err = extract_json_from_text(raw_json, auto_repair=True)
        
        if parse_err:
            chunk_debug["status"] = "parse_error"
            chunk_debug["error"] = parse_err
            return None, chunk_debug
        
        if not data:
            chunk_debug["status"] = "no_data"
            return None, chunk_debug
        
        df = pd.DataFrame(data)
        cols = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
        df = df.rename(columns=cols)
        for c in cols.values():
            if c not in df.columns:
                df[c] = ""
        
        chunk_debug["status"] = "success"
        return df, chunk_debug

    @staticmethod
    def process_image(filename, image_bytes, api_key):
        """处理图片 (纯函数)"""
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
                max_tokens=2048
            )
            debug_log["steps"].append(f"视觉模型耗时: {time.time()-t1:.4f}s")
            
            raw_json = resp.choices[0].message.content
            debug_log["ai_response_full"] = raw_json
            
            data, parse_err = extract_json_from_text(raw_json, auto_repair=True)
            
            if parse_err:
                 return None, f"识别结果解析失败: {parse_err}", debug_log

            if isinstance(data, dict): data = [data]
            
            if not data: return None, "未识别到数据", debug_log
            
            df = pd.DataFrame(data)
            # 简单映射
            cols = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
            df = df.rename(columns=cols)
            for c in cols.values(): 
                if c not in df.columns: df[c] = ""
            
            return df, None, debug_log
            
        except Exception as e:
            return None, str(e), debug_log

    @staticmethod
    def merge_data(old_df, new_df):
        """合并去重"""
        if new_df is None or new_df.empty: return old_df, 0
        
        # 简单指纹
        def get_fp(d): 
            return d['日期'].astype(str) + d['金额'].astype(str) + d['备注'].str[:5]
            
        if old_df.empty: 
            return new_df, len(new_df)
            
        old_fp = set(get_fp(old_df))
        new_df['_fp'] = get_fp(new_df)
        
        to_add = new_df[~new_df['_fp'].isin(old_fp)].drop(columns=['_fp'])
        
        if to_add.empty: return old_df, 0
        
        merged = pd.concat([old_df, to_add], ignore_index=True)
        # 确保合并后清洗类型，防止后续报错
        merged = DataManager._clean_df_types(merged)
        merged = merged.sort_values('日期', ascending=False).reset_index(drop=True)
        return merged, len(to_add)

# --- 主程序 ---
def main():
    # 1. 初始化与侧边栏
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

    # 2. 数据加载
    if 'ledger_data' not in st.session_state:
        df, sha = dm.load_data()
        st.session_state.ledger_data = df
        st.session_state.github_sha = sha

    # 3. 顶部概览
    st.title("💰 AI 智能账本 Pro")
    
    today = date.today()
    # 简单的账期计算
    target_month = today.month if today.day < payday else (today.month % 12) + 1
    target_year = today.year + (1 if (today.month==12 and today.day >= payday) else 0)
    target_date = date(target_year, target_month, payday)
    days_left = (target_date - today).days

    df = st.session_state.ledger_data.copy()
    month_spend = 0.0
    if not df.empty:
        # 确保类型安全
        if '日期' not in df.columns:
             df['日期'] = []
        df['dt'] = pd.to_datetime(df['日期'], errors='coerce')
        # 本月支出 (自然月)
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

    # 4. 主要功能区
    t_import, t_add, t_history, t_stats = st.tabs(["📥 智能导入", "✍️ 手动记账", "📋 历史明细", "📊 统计报表"])

    # --- 智能导入 Tab ---
    with t_import:
        files = st.file_uploader("支持 PDF/CSV/Excel/图片", accept_multiple_files=True)
        if files and st.button("🚀 批量开始识别", type="primary"):
            if not api_key:
                st.error("缺少 API Key")
                st.stop()
            
            # 1. 预读取所有文件 (避免在线程中传 Streamlit 对象)
            tasks_doc = []
            tasks_img = []
            
            with st.status("正在预处理文件...") as status:
                for f in files:
                    ext = f.name.split('.')[-1].lower()
                    f.seek(0) # 关键：重置指针
                    bytes_data = f.read()
                    
                    item = {"name": f.name, "bytes": bytes_data}
                    if ext in ['png', 'jpg', 'jpeg']:
                        tasks_img.append(item)
                    else:
                        tasks_doc.append(item)
                status.update(label="文件读取完成，准备提交 AI", state="complete")

            # 2. 并发处理
            new_df = pd.DataFrame()
            debug_logs = []
            
            progress = st.progress(0)
            total_tasks = len(tasks_doc) + len(tasks_img)
            completed = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {}
                
                # 提交文档任务
                for t in tasks_doc:
                    f = executor.submit(BillParser.identify_and_parse, t['name'], t['bytes'], api_key)
                    futures[f] = t['name']
                
                # 提交图片任务
                for t in tasks_img:
                    f = executor.submit(BillParser.process_image, t['name'], t['bytes'], api_key)
                    futures[f] = t['name']
                
                # 等待结果
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

            # 3. 显示调试信息
            if st.session_state.debug_mode:
                with st.expander("🔬 深度调试信息 (包含分片详情)", expanded=True):
                    # 显示整体日志
                    st.json(debug_logs)
                    
                    # 显示每个分片的详细情况
                    for dbg in debug_logs:
                        if "chunks" in dbg:
                            for chunk_info in dbg["chunks"]:
                                with st.container():
                                    col1, col2 = st.columns([1, 4])
                                    status_emoji = "✅" if chunk_info.get("status") == "success" else "❌"
                                    col1.markdown(f"**分片 {chunk_info.get('index', 0) + 1}** {status_emoji}")
                                    
                                    if chunk_info.get("status") == "success":
                                        col2.caption(f"提取记录: {chunk_info.get('records', 0)} 条")
                                        # 可展开查看原始响应
                                        if "debug" in chunk_info and "ai_response_full" in chunk_info["debug"]:
                                            with col2.expander("查看原始响应"):
                                                st.code(chunk_info["debug"]["ai_response_full"], language="json")
                                    else:
                                        col2.error(f"{chunk_info.get('error', '未知错误')}")

            # 4. 保存
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
                            st.error("保存失败，请检查网络或配置")
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
            
            if st.form_submit_button("💾 保存", width="stretch"):
                row = pd.DataFrame([{"日期": str(d), "类型": t, "金额": a, "分类": cat, "备注": rem}])
                merged, added = DataManager.merge_data(st.session_state.ledger_data, row)
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
            # 确保在展示前数据是干净的
            st.session_state.ledger_data = DataManager._clean_df_types(st.session_state.ledger_data)
            
            edited_df = st.data_editor(
                st.session_state.ledger_data,
                use_container_width=True,
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
            
            if st.button("💾 保存表格变更"):
                if not edited_df.equals(st.session_state.ledger_data):
                    with st.spinner("同步中..."):
                        ok, new_sha = dm.save_data(edited_df, st.session_state.get('github_sha'))
                        if ok:
                            st.session_state.ledger_data = edited_df
                            st.session_state.github_sha = new_sha
                            st.success("✅ 更新成功")
                else:
                    st.info("数据未变更")

    # --- 统计报表 Tab ---
    with t_stats:
        if st.session_state.ledger_data.empty:
            st.info("暂无数据，请先记账")
        else:
            df = st.session_state.ledger_data.copy()
            # 类型转换用于绘图
            df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
            
            # 筛选
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

            # AI 分析模块
            st.divider()
            st.subheader("🤖 AI 财务顾问")
            if st.button("生成本月分析报告"):
                if not api_key:
                    st.error("请配置 API Key")
                else:
                    with st.spinner("AI 正在分析您的财务状况..."):
                        # 仅发送最近 100 条支出数据，避免超长
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
                            st.error(f"AI 分析失败: {e}")

if __name__ == "__main__":
    main()
