# import os
# from glob import glob
# from PIL import Image
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from flask import Flask, request, jsonify
# import json
# import sys
# import io
# import requests
# import zipfile
# import logging


# # 安裝必要的套件（請先在終端執行這些命令）
# # pip install sentence-transformers
# # pip install faiss-cpu
# # pip install requests

# # 設定日誌
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # 設定 UTF-8 編碼以顯示中文
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# # # 確保工作目錄為專案根目錄
# # os.chdir(os.path.dirname(os.path.abspath(__file__)))

# # 初始化 Flask 應用
# app = Flask(__name__)

# # 下載並解壓檔案
# def download_and_extract_zip(url, output_zip="data.zip", extract_to="."):
#     print("正在下載檔案...")
#     response = requests.get(url, stream=True)
#     if response.status_code == 200:
#         with open(output_zip, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         print("下載完成，開始解壓...")
#         with zipfile.ZipFile(output_zip, 'r') as zip_ref:
#             zip_ref.extractall(extract_to)
#         os.remove(output_zip)  # 移除臨時 zip 檔案
#         print("解壓完成")
#     else:
#         raise Exception(f"下載失敗，狀態碼: {response.status_code}")

# # # 下載資料
# # url = "https://drive.usercontent.google.com/download?id=1rVmGMzxnxs5twJ27A4TeP1TDZb4x3pOF&export=download&authuser=0"
# # download_and_extract_zip(url)

# # 設定資料庫圖片路徑（使用 Render 的持久化磁碟）
# DATABASE_PATH = os.path.join(os.getcwd(), "medicine_images")  # 資料庫圖片資料夾
# OUTPUT_INDEX_PATH = os.path.join(os.getcwd(), "vector.index")
# QUERY_PATH = os.path.join(os.getcwd(), "query_images")  # 查詢圖片資料夾

# # 下載圖片
# if not os.path.exists(DATABASE_PATH) or not glob(os.path.join(DATABASE_PATH, "*.jpg")):
#     os.makedirs(DATABASE_PATH, exist_ok=True)
#     download_and_extract_zip(
#         "https://drive.usercontent.google.com/download?id=1rVmGMzxnxs5twJ27A4TeP1TDZb4x3pOF&export=download&authuser=0",
#         extract_to=DATABASE_PATH
#     )

# # 如果查詢資料夾不存在，創建一個並放入範例查詢圖片（這裡假設手動準備）
# if not os.path.exists(QUERY_PATH):
#     os.makedirs(QUERY_PATH)
#     print(f"已創建查詢資料夾: {QUERY_PATH}，請將查詢圖片放入此資料夾")

# # 獲取資料庫圖片檔案（不包含查詢圖片）
# database_files = glob(os.path.join(DATABASE_PATH, '*.jpg'))
# print(f"資料庫中找到 {len(database_files)} 張圖片：{[os.path.basename(f) for f in database_files]}")

# # # 顯示資料庫中的圖片
# # plt.figure(figsize=(20, max(2, len(database_files) // 5 + 1) * 4))
# # for i, image_path in enumerate(database_files):
# #     img = Image.open(image_path)
# #     plt.subplot((len(database_files) + 4) // 5, 5, i + 1)
# #     plt.imshow(img)
# #     plt.title(os.path.basename(image_path))
# #     plt.axis('off')
# # plt.tight_layout()
# # plt.show()

# # 定義生成 CLIP 嵌入的函數
# def generate_clip_embeddings(images_path, model):
#     image_paths = glob(os.path.join(images_path, '*.jpg'))  # 僅頂層
#     if not image_paths:
#         raise Exception(f"No images found in {images_path}")
#     embeddings = []
#     file_names = []
    
#     print(f"正在為資料庫生成嵌入，找到 {len(image_paths)} 張圖片")
#     for img_path in image_paths:
#         try:
#             image = Image.open(img_path).convert('RGB')  # 確保 RGB 格式
#             embedding = model.encode(image, show_progress_bar=False)
#             embeddings.append(embedding)
#             file_names.append(os.path.basename(img_path))
#             print(f"已處理: {os.path.basename(img_path)}")
#         except Exception as e:
#             print(f"處理 {img_path} 時出錯: {e}")
#             continue  # 跳過錯誤圖片
    
#     if not embeddings:
#         raise Exception("No valid embeddings generated")
#     return embeddings, image_paths, file_names

# # 設定模型
# # IMAGES_PATH = os.path.join(os.getcwd(), "medicine_images")
# # model = SentenceTransformer('clip-ViT-B-32')
# # embeddings, image_paths, file_names = generate_clip_embeddings(DATABASE_PATH, model)

# # 創建 FAISS 索引並儲存 metadata
# def create_faiss_index(embeddings, image_paths, file_names, output_path):
#     if not embeddings:
#         raise Exception("No embeddings provided for indexing")
#     dimension = len(embeddings[0])
#     index = faiss.IndexFlatIP(dimension)
#     index = faiss.IndexIDMap(index)
    
#     vectors = np.array(embeddings).astype(np.float32)
#     index.add_with_ids(vectors, np.array(range(len(embeddings))))
    
#     faiss.write_index(index, output_path)
#     print(f"索引已保存至 {output_path}")
    
#     # 儲存 metadata，包括中文名稱
#     metadata = []
#     metadata_path = output_path + '.metadata.json'

#     # # 如果已有元數據檔案，讀取現有資料
#     # if os.path.exists(metadata_path):
#     #     with open(metadata_path, 'r', encoding='utf-8') as f:
#     #         existing_metadata = json.load(f)
#     #     existing_dict = {item["file_name"]: item for item in existing_metadata}
#     # else:
#     #     existing_dict = {}

#     # 假設你有一個外部來源的藥品資訊字典（範例）
#     medicine_info = {
#         "福善美保骨錠-1.jpg": {"medicationCode":"1AMZ08", "genericName":"Alendronate", "chineseBrandName": "福善美保骨錠","englishBrandName": "Fosamax PLUS"},
#         "福善美保骨錠-2.jpg": {"medicationCode":"1AMZ08", "genericName":"Alendronate", "chineseBrandName": "福善美保骨錠","englishBrandName": "Fosamax PLUS"},
#         "芙琳亞錠-1.jpg": {"medicationCode":"1MAC12", "genericName":"Calcium Folinate", "chineseBrandName": "芙琳亞錠","englishBrandName": "Folina"},
#         "芙琳亞錠-2.jpg": {"medicationCode":"1MAC12", "genericName":"Calcium Folinate", "chineseBrandName": "芙琳亞錠","englishBrandName": "Folina"},        
#         "達滋克膜衣錠-1.jpg": {"medicationCode":"1MBD06", "genericName":"Lamivudine/Tenofovir/Doravirine", "chineseBrandName": "達滋克膜衣錠","englishBrandName": "FDelstrigo"},
#         "達滋克膜衣錠-2.jpg": {"medicationCode":"1MBD06", "genericName":"Lamivudine/Tenofovir/Doravirine", "chineseBrandName": "達滋克膜衣錠","englishBrandName": "FDelstrigo"},
#         "敵芬尼朵糖衣錠-1.jpg": {"medicationCode":"1MAD01", "genericName":"Diphenidol HCl", "chineseBrandName": "敵芬妮朵糖衣錠","englishBrandName": "Diphenidol"},
#         "解鐵定膜衣錠-1.jpg": {"medicationCode":"1MAD07", "genericName":"Deferasirox", "chineseBrandName": "解鐵定膜衣錠","englishBrandName": "Jadenu"},
#         "解鐵定膜衣錠-2.jpg": {"medicationCode":"1MAD07", "genericName":"Deferasirox", "chineseBrandName": "解鐵定膜衣錠","englishBrandName": "Jadenu"},
#         "佩你安錠-1.jpg": {"medicationCode":"1MAC08", "genericName":"Cyproheptadine HCl", "chineseBrandName": "佩你安錠","englishBrandName": "Pilian"},
#         "佩你安錠-2.jpg": {"medicationCode":"1MAC08", "genericName":"Cyproheptadine HCl", "chineseBrandName": "佩你安錠","englishBrandName": "Pilian"},
#         "法瑪鎮膜衣錠-1.jpg": {"medicationCode":"1MAF07", "genericName":"Famotidine", "chineseBrandName": "法瑪鎮膜衣錠","englishBrandName": "Famotidine"},
#         "法瑪鎮膜衣錠-2.jpg": {"medicationCode":"1MAF07", "genericName":"Famotidine", "chineseBrandName": "法瑪鎮膜衣錠","englishBrandName": "Famotidine"},
#         "睦體康腸衣錠-1.jpg": {"medicationCode":"1AMZ07", "genericName":"Mycophenolate Sodium", "chineseBrandName": "睦體康腸衣錠","englishBrandName": "Myfortic"},
#         "睦體康腸衣錠-2.jpg": {"medicationCode":"1AMZ07", "genericName":"Mycophenolate Sodium", "chineseBrandName": "睦體康腸衣錠","englishBrandName": "Myfortic"},
#         "樂伯克錠-1.jpg": {"medicationCode":"1AMG21", "genericName":"Pramipexole", "chineseBrandName": "樂伯克錠","englishBrandName": "Mirapex"},
#         "樂伯克錠-2.jpg": {"medicationCode":"1AMG21", "genericName":"Pramipexole", "chineseBrandName": "樂伯克錠","englishBrandName": "Mirapex"},
#         "諾博戈膜衣錠-1.jpg": {"medicationCode":"1MDD09", "genericName":"Darolutamide", "chineseBrandName": "諾博戈膜衣錠","englishBrandName": "Nubeqa"},
#         "諾博戈膜衣錠-2.jpg": {"medicationCode":"1MDD09", "genericName":"Darolutamide", "chineseBrandName": "諾博戈膜衣錠","englishBrandName": "Nubeqa"}
#         # "福善美保骨錠-1.jpg": {"genericName":"", "chineseBrandName": "","englishBrandName": ""},
#         # "福善美保骨錠-1.jpg": {"genericName":"", "chineseBrandName": "","englishBrandName": ""},
#     }

#     # 更新或創建元數據
#     for img_path, file_name in zip(image_paths, file_names):
#         info = medicine_info.get(file_name, {"medicationCode": "UNKNOWN", "chineseBrandName": "未知藥品"})
#         metadata.append({
#             "file_name": file_name,
#             "full_path": img_path,
#             "additional_info": info
#         })
    
#     with open(metadata_path, 'w', encoding='utf-8') as f:
#         json.dump(metadata, f, indent=4)
    
#     print(f"索引和元數據已保存至 {output_path}")
#     return index

# # 讀取 FAISS 索引與 metadata
# def load_faiss_index(index_path):
#     try:
#         index = faiss.read_index(index_path)
#         with open(index_path + '.metadata.json', 'r', encoding='utf-8') as f:
#             metadata = json.load(f)
#         metadata_dict = {item["file_name"]: item for item in metadata}
#         print(f"索引已從 {index_path} 載入")
#         return index, metadata_dict
#     except Exception as e:
#         raise Exception(f"載入索引失敗: {e}")

# # 載入模型和索引
# model = SentenceTransformer('clip-ViT-B-32')

# # 如果索引不存在，則生成
# if not os.path.exists(OUTPUT_INDEX_PATH):
#     embeddings, image_paths, file_names = generate_clip_embeddings(DATABASE_PATH, model)
#     index = create_faiss_index(embeddings, image_paths, file_names, OUTPUT_INDEX_PATH)
#     index, metadata_dict = load_faiss_index(OUTPUT_INDEX_PATH)    # 新建完之後再載入 metadata_dict
# else:
#     index, metadata_dict = load_faiss_index(OUTPUT_INDEX_PATH)

# # 檢索相似圖片
# def retrieve_similar_images(query, model, index, metadata_dict, top_k=1):  # 只返回最相似的結果
#     # 圖片查詢
#     try:
#         if isinstance(query, str):
#             query = Image.open(query)
#         query_features = model.encode(query)
#         query_features = query_features.astype(np.float32).reshape(1, -1)
#         distances, indices = index.search(query_features, top_k)
#         retrieved_metadata = [metadata_dict[list(metadata_dict.keys())[int(idx)]] for idx in indices[0]]

#         # 從 metadata 中返回最相似的藥品的中文名稱
#         if retrieved_metadata:
#             medication_code = retrieved_metadata[0]["additional_info"]["medicationCode"]
#             chinese_name = retrieved_metadata[0]["additional_info"]["chineseBrandName"]
#             return medication_code, chinese_name
#         return None, None
#     except Exception as e:
#         print(f"檢索圖片時出錯: {e}")
#         return None, None

# # API 端點：接收圖片並返回中文名稱
# @app.route('/query_image', methods=['POST'])
# def query_image():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image provided"}), 400
    
#     file = request.files['image']
#     try:
#         query_img = Image.open(file.stream).convert('RGB')
#         medication_code, chinese_name = retrieve_similar_images(query_img, model, index, metadata_dict)
        
#         if medication_code and chinese_name:
#             return jsonify({"medicationCode": medication_code, "chineseBrandName": chinese_name})
#         return jsonify({"error": "No match found"}), 404
#     except Exception as e:
#         print(f"處理查詢圖片時出錯: {e}")
#         return jsonify({"error": "Invalid image"}), 400

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 8000))  # Render 預設端口
#     app.run(host='127.0.0.1', port=port)


# # # 可視化結果（僅支援圖片查詢）
# # def visualize_results(query, retrieved_metadata):
# #     plt.figure(figsize=(15, 5))
    
# #     plt.subplot(1, len(retrieved_metadata) + 1, 1)
# #     plt.imshow(query)
# #     plt.title("查詢圖片")
# #     plt.axis('off')
    
# #     for i, metadata in enumerate(retrieved_metadata):
# #         img_path = metadata["full_path"]
# #         info = metadata["additional_info"]
# #         if isinstance(info, dict):
# #             title = f"匹配 {i + 1}\n中文名: {info.get('chineseBrandName', '未知')}\n英文名: {info.get('englishBrandName', '無')}\n學名: {info.get('genericName', '未知')}"
# #         else:
# #             title = f"匹配 {i + 1}\n資訊: {info if info else '無'}"
# #         plt.subplot(1, len(retrieved_metadata) + 1, i + 2)
# #         plt.imshow(Image.open(img_path))
# #         plt.title(title)
# #         plt.axis('off')
    
# #     plt.tight_layout()
# #     plt.show()

# # # 測試圖片查詢
# # print("測試圖片查詢")
# # query_image = os.path.join(QUERY_PATH, "諾博戈膜衣錠-2.jpg")  
# # query, retrieved_metadata = retrieve_similar_images(query_image, model, index, metadata_dict)
# # visualize_results(query, retrieved_metadata)













# import os
# from glob import glob
# from PIL import Image
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from flask import Flask, request, jsonify
# import json
# import sys
# import io
# import requests
# import zipfile
# import logging
# from waitress import serve

# # 設定日誌
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # 設定 UTF-8 編碼
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# # 初始化 Flask 應用
# app = Flask(__name__)

# # 確保工作目錄
# try:
#     os.chdir(os.path.dirname(os.path.abspath(__file__)))
#     logger.info(f"工作目錄設為: {os.getcwd()}")
# except Exception as e:
#     logger.error(f"設定工作目錄失敗: {e}")
#     sys.exit(1)

# # 下載並解壓檔案
# def download_and_extract_zip(url, output_zip="data.zip", extract_to="."):
#     logger.info("正在下載檔案...")
#     try:
#         response = requests.get(url, stream=True, timeout=30)
#         response.raise_for_status()
#         with open(output_zip, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         logger.info("下載完成，開始解壓...")
#         with zipfile.ZipFile(output_zip, 'r') as zip_ref:
#             zip_ref.extractall(extract_to)
#         os.remove(output_zip)
#         logger.info("解壓完成")
#     except Exception as e:
#         logger.error(f"下載或解壓失敗: {e}")
#         raise

# # 設定路徑
# DATABASE_PATH = os.path.join(os.getcwd(), "medicine_images")
# OUTPUT_INDEX_PATH = os.path.join(os.getcwd(), "vector.index")
# QUERY_PATH = os.path.join(os.getcwd(), "query_images")

# # 初始化全局變數
# model = None
# index = None
# metadata_dict = None

# # 定義生成 CLIP 嵌入的函數
# def generate_clip_embeddings(images_path, model):
#     image_paths = glob(os.path.join(images_path, '*.jpg'))
#     if not image_paths:
#         raise Exception(f"No images found in {images_path}")
#     embeddings = []
#     file_names = []
    
#     logger.info(f"正在為資料庫生成嵌入，找到 {len(image_paths)} 張圖片")
#     for img_path in image_paths:
#         try:
#             image = Image.open(img_path).convert('RGB')
#             embedding = model.encode(image, show_progress_bar=False)
#             embeddings.append(embedding)
#             file_names.append(os.path.basename(img_path))
#             logger.info(f"已處理: {os.path.basename(img_path)}")
#         except Exception as e:
#             logger.error(f"處理 {img_path} 時出錯: {e}")
#             continue
    
#     if not embeddings:
#         raise Exception("No valid embeddings generated")
#     return embeddings, image_paths, file_names

# # 創建 FAISS 索引
# def create_faiss_index(embeddings, image_paths, file_names, output_path):
#     if not embeddings:
#         raise Exception("No embeddings provided for indexing")
#     dimension = len(embeddings[0])
#     index = faiss.IndexFlatIP(dimension)
#     index = faiss.IndexIDMap(index)
    
#     vectors = np.array(embeddings).astype(np.float32)
#     index.add_with_ids(vectors, np.array(range(len(embeddings))))
    
#     faiss.write_index(index, output_path)
#     logger.info(f"索引已保存至 {output_path}")
    
#     metadata = []
#     metadata_path = output_path + '.metadata.json'
#     medicine_info = {
#         "福善美保骨錠-1.jpg": {"medicationCode":"1AMZ08", "genericName":"Alendronate", "chineseBrandName": "福善美保骨錠","englishBrandName": "Fosamax PLUS"},
#         "福善美保骨錠-2.jpg": {"medicationCode":"1AMZ08", "genericName":"Alendronate", "chineseBrandName": "福善美保骨錠","englishBrandName": "Fosamax PLUS"},
#         "芙琳亞錠-1.jpg": {"medicationCode":"1MAC12", "genericName":"Calcium Folinate", "chineseBrandName": "芙琳亞錠","englishBrandName": "Folina"},
#         "芙琳亞錠-2.jpg": {"medicationCode":"1MAC12", "genericName":"Calcium Folinate", "chineseBrandName": "芙琳亞錠","englishBrandName": "Folina"},
#         "達滋克膜衣錠-1.jpg": {"medicationCode":"1MBD06", "genericName":"Lamivudine/Tenofovir/Doravirine", "chineseBrandName": "達滋克膜衣錠","englishBrandName": "FDelstrigo"},
#         "達滋克膜衣錠-2.jpg": {"medicationCode":"1MBD06", "genericName":"Lamivudine/Tenofovir/Doravirine", "chineseBrandName": "達滋克膜衣錠","englishBrandName": "FDelstrigo"},
#         "敵芬妮朵糖衣錠-1.jpg": {"medicationCode":"1MAD01", "genericName":"Diphenidol HCl", "chineseBrandName": "敵芬妮朵糖衣錠","englishBrandName": "Diphenidol"},
#         "解鐵定膜衣錠-1.jpg": {"medicationCode":"1MAD07", "genericName":"Deferasirox", "chineseBrandName": "解鐵定膜衣錠","englishBrandName": "Jadenu"},
#         "解鐵定膜衣錠-2.jpg": {"medicationCode":"1MAD07", "genericName":"Deferasirox", "chineseBrandName": "解鐵定膜衣錠","englishBrandName": "Jadenu"},
#         "佩你安錠-1.jpg": {"medicationCode":"1MAC08", "genericName":"Cyproheptadine HCl", "chineseBrandName": "佩你安錠","englishBrandName": "Pilian"},
#         "佩你安錠-2.jpg": {"medicationCode":"1MAC08", "genericName":"Cyproheptadine HCl", "chineseBrandName": "佩你安錠","englishBrandName": "Pilian"},
#         "法瑪鎮膜衣錠-1.jpg": {"medicationCode":"1MAF07", "genericName":"Famotidine", "chineseBrandName": "法瑪鎮膜衣錠","englishBrandName": "Famotidine"},
#         "法瑪鎮膜衣錠-2.jpg": {"medicationCode":"1MAF07", "genericName":"Famotidine", "chineseBrandName": "法瑪鎮膜衣錠","englishBrandName": "Famotidine"},
#         "睦體康腸衣錠-1.jpg": {"medicationCode":"1AMZ07", "genericName":"Mycophenolate Sodium", "chineseBrandName": "睦體康腸衣錠","englishBrandName": "Myfortic"},
#         "睦體康腸衣錠-2.jpg": {"medicationCode":"1AMZ07", "genericName":"Mycophenolate Sodium", "chineseBrandName": "睦體康腸衣錠","englishBrandName": "Myfortic"},
#         "樂伯克錠-1.jpg": {"medicationCode":"1AMG21", "genericName":"Pramipexole", "chineseBrandName": "樂伯克錠","englishBrandName": "Mirapex"},
#         "樂伯克錠-2.jpg": {"medicationCode":"1AMG21", "genericName":"Pramipexole", "chineseBrandName": "樂伯克錠","englishBrandName": "Mirapex"},
#         "諾博戈膜衣錠-1.jpg": {"medicationCode":"1MDD09", "genericName":"Darolutamide", "chineseBrandName": "諾博戈膜衣錠","englishBrandName": "Nubeqa"},
#         "諾博戈膜衣錠-2.jpg": {"medicationCode":"1MDD09", "genericName":"Darolutamide", "chineseBrandName": "諾博戈膜衣錠","englishBrandName": "Nubeqa"}
#     }
#     for img_path, file_name in zip(image_paths, file_names):
#         info = medicine_info.get(file_name, {"medicationCode": "UNKNOWN", "chineseBrandName": "未知藥品"})
#         metadata.append({
#             "file_name": file_name,
#             "full_path": img_path,
#             "additional_info": info
#         })
    
#     with open(metadata_path, 'w', encoding='utf-8') as f:
#         json.dump(metadata, f, indent=4)
    
#     logger.info(f"元數據已保存至 {metadata_path}")
#     return index

# # 讀取 FAISS 索引
# def load_faiss_index(index_path):
#     try:
#         index = faiss.read_index(index_path)
#         with open(index_path + '.metadata.json', 'r', encoding='utf-8') as f:
#             metadata = json.load(f)
#         metadata_dict = {item["file_name"]: item for item in metadata}
#         logger.info(f"索引已從 {index_path} 載入")
#         return index, metadata_dict
#     except Exception as e:
#         logger.error(f"載入索引失敗: {e}")
#         raise

# # 初始化模型和索引
# def initialize():
#     global model, index, metadata_dict
#     try:
#         logger.info("正在載入模型...")
#         model = SentenceTransformer('clip-ViT-B-32', device='cpu')
#         logger.info("模型載入完成")
        
#         if not os.path.exists(DATABASE_PATH) or not glob(os.path.join(DATABASE_PATH, "*.jpg")):
#             os.makedirs(DATABASE_PATH, exist_ok=True)
#             download_and_extract_zip(
#                 "https://drive.usercontent.google.com/download?id=1rVmGMzxnxs5twJ27A4TeP1TDZb4x3pOF&export=download&authuser=0",
#                 extract_to=DATABASE_PATH
#             )
        
#         if not os.path.exists(QUERY_PATH):
#             os.makedirs(QUERY_PATH)
#             logger.info(f"已創建查詢資料夾: {QUERY_PATH}")
        
#         database_files = glob(os.path.join(DATABASE_PATH, '*.jpg'))
#         logger.info(f"資料庫中找到 {len(database_files)} 張圖片：{[os.path.basename(f) for f in database_files]}")
        
#         if not os.path.exists(OUTPUT_INDEX_PATH):
#             logger.info("索引不存在，正在生成...")
#             embeddings, image_paths, file_names = generate_clip_embeddings(DATABASE_PATH, model)
#             index = create_faiss_index(embeddings, image_paths, file_names, OUTPUT_INDEX_PATH)
#             index, metadata_dict = load_faiss_index(OUTPUT_INDEX_PATH)
#         else:
#             index, metadata_dict = load_faiss_index(OUTPUT_INDEX_PATH)
#     except Exception as e:
#         logger.error(f"初始化失敗: {e}")
#         sys.exit(1)

# # 檢索相似圖片
# def retrieve_similar_images(query, model, index, metadata_dict, top_k=1):
#     try:
#         if isinstance(query, str):
#             query = Image.open(query)
#         query_features = model.encode(query)
#         query_features = query_features.astype(np.float32).reshape(1, -1)
#         distances, indices = index.search(query_features, top_k)
#         retrieved_metadata = [metadata_dict[list(metadata_dict.keys())[int(idx)]] for idx in indices[0]]
        
#         if retrieved_metadata:
#             medication_code = retrieved_metadata[0]["additional_info"]["medicationCode"]
#             chinese_name = retrieved_metadata[0]["additional_info"]["chineseBrandName"]
#             return medication_code, chinese_name
#         return None, None
#     except Exception as e:
#         logger.error(f"檢索圖片時出錯: {e}")
#         return None, None

# # 測試路由
# @app.route('/test', methods=['GET'])
# def test():
#     return jsonify({"message": "Server is running"}), 200

# # API 端點
# @app.route('/query_image', methods=['POST'])
# def query_image():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image provided"}), 400
    
#     file = request.files['image']
#     try:
#         query_img = Image.open(file.stream).convert('RGB')
#         medication_code, chinese_name = retrieve_similar_images(query_img, model, index, metadata_dict)
        
#         if medication_code and chinese_name:
#             return jsonify({"medicationCode": medication_code, "chineseBrandName": chinese_name})
#         return jsonify({"error": "No match found"}), 404
#     except Exception as e:
#         logger.error(f"處理查詢圖片時出錯: {e}")
#         return jsonify({"error": "Invalid image"}), 400

# # 初始化並啟動
# if __name__ == '__main__':
#     initialize()
#     try:
#         port = int(os.environ.get("PORT", 5000))
#         logger.info(f"啟動 Waitress 服務於 http://127.0.0.1:{port}")
#         serve(app, host='0.0.0.0', port=port, threads=4)
#     except Exception as e:
#         logger.error(f"啟動服務失敗: {e}")
#         sys.exit(1)















# #沒有下載邏輯的版本
# # 這個版本假設資料庫已經存在於指定的路徑中，並不會自動下載資料庫
# #優化版
# import os
# from glob import glob
# from PIL import Image
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from flask import Flask, request, jsonify
# import json
# import sys
# import io
# import logging
# from waitress import serve

# # 設定日誌
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # 設定 UTF-8 編碼
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# # 初始化 Flask 應用
# app = Flask(__name__)

# # 確保工作目錄
# try:
#     os.chdir(os.path.dirname(os.path.abspath(__file__)))
#     logger.info(f"工作目錄設為: {os.getcwd()}")
# except Exception as e:
#     logger.error(f"設定工作目錄失敗: {e}", exc_info=True)
#     sys.exit(1)

# # 設定路徑
# OUTPUT_INDEX_PATH = os.path.join(os.getcwd(), "vector.index")
# IMAGE_DIR = os.path.join(os.getcwd(), "medicine_images", "compressed")

# # 初始化全局變數
# index = None
# metadata_dict = None
# model = None

# # 讀取 FAISS 索引
# def load_faiss_index(index_path):
#     try:
#         index = faiss.read_index(index_path)
#         with open(index_path + '.metadata.json', 'r', encoding='utf-8') as f:
#             metadata = json.load(f)
#         metadata_dict = {item["file_name"]: item for item in metadata}
#         logger.info(f"索引已從 {index_path} 載入，元數據鍵: {list(metadata_dict.keys())}")
#         return index, metadata_dict
#     except Exception as e:
#         logger.error(f"載入索引失敗: {e}", exc_info=True)
#         raise

# # 初始化
# def initialize():
#     global index, metadata_dict, model
#     try:
#         if not os.path.exists(OUTPUT_INDEX_PATH):
#             logger.error("索引檔案不存在，請預先生成")
#             sys.exit(1)
#         if not os.path.exists(OUTPUT_INDEX_PATH + '.metadata.json'):
#             logger.error("元數據檔案不存在，請預先生成")
#             sys.exit(1)
#         index, metadata_dict = load_faiss_index(OUTPUT_INDEX_PATH)
#         model = SentenceTransformer('clip-ViT-B-32', device='cpu')
#         logger.debug("模型 clip-ViT-B-32 載入完成")
#     except Exception as e:
#         logger.error(f"初始化失敗: {e}", exc_info=True)
#         sys.exit(1)
#     #     if not os.path.exists(OUTPUT_INDEX_PATH):
#     #         logger.error("索引檔案不存在，請預先生成")
#     #         sys.exit(1)
#     #     index, metadata_dict = load_faiss_index(OUTPUT_INDEX_PATH)
#     #     model = SentenceTransformer('clip-ViT-B-32', device='cpu')
#     #     logger.debug("模型 clip-ViT-B-32 載入完成")        
#     # except Exception as e:
#     #     logger.error(f"初始化失敗: {e}", exc_info=True)
#     #     sys.exit(1)
    

# # 檢索相似圖片
# def retrieve_similar_images(query, metadata_dict, top_k=1):
#     global model
#     try:
#         if isinstance(query, str):
#             query = Image.open(query).convert('RGB')
#         query_features = model.encode(query, show_progress_bar=False)
#         query_features = query_features.astype(np.float32).reshape(1, -1)
#         distances, indices = index.search(query_features, top_k)
#         logger.debug(f"檢索結果: 距離={distances}, 索引={indices}")
#         retrieved_metadata = [metadata_dict[list(metadata_dict.keys())[int(idx)]] for idx in indices[0]]
        
#         if retrieved_metadata:
#             medication_code = retrieved_metadata[0]["additional_info"]["medicationCode"]
#             chinese_name = retrieved_metadata[0]["additional_info"]["chineseBrandName"]
#             logger.debug(f"匹配成功: {chinese_name} ({medication_code})")
#             return medication_code, chinese_name
#         logger.warning("未找到匹配圖片")
#         return None, None
#     except Exception as e:
#         logger.error(f"檢索圖片時出錯: {e}", exc_info=True)
#         return None, None
#     finally:
#         if isinstance(query, Image.Image):
#             query.close()

# # 測試路由
# @app.route('/test', methods=['GET'])
# def test():
#     logger.debug("收到 /test 請求")
#     return jsonify({"message": "Server is running"}), 200

# # API 端點
# @app.route('/query_image', methods=['POST'])
# def query_image():    
#     logger.debug("收到 /query_image 請求")
#     if 'image' not in request.files:
#         logger.error("未提供圖片")
#         return jsonify({"error": "No image provided"}), 400
    
#     file = request.files['image']
#     try:
#         if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#             logger.error(f"不支援的檔案格式: {file.filename}")
#             return jsonify({"error": "Unsupported image format. Use JPG or PNG"}), 400
        
#         file.stream.seek(0)
#         query_img = Image.open(file.stream).convert('RGB')
#         logger.debug("圖片成功載入")
#         medication_code, chinese_name = retrieve_similar_images(query_img, metadata_dict)
        
#         if medication_code and chinese_name:
#             logger.debug(f"回應: medicationCode={medication_code}, chineseBrandName={chinese_name}")
#             return jsonify({"medicationCode": medication_code, "chineseBrandName": chinese_name})
#         return jsonify({"error": "No match found"}), 404
#     except Exception as e:
#         logger.error(f"處理查詢圖片時出錯: {e}", exc_info=True)
#         return jsonify({"error": "Invalid image", "details": str(e)}), 400
#     finally:
#         if 'query_img' in locals():
#             query_img.close()
#             logger.debug("圖片資源已釋放")

# # def query_image():
# #     logger.debug("收到 /query_image 請求")
# #     if 'image' not in request.files:
# #         logger.error("未提供圖片")
# #         return jsonify({"error": "No image provided"}), 400
    
# #     file = request.files['image']
# #     try:
# #         query_img = Image.open(file.stream).convert('RGB')
# #         medication_code, chinese_name, model = retrieve_similar_images(query_img, metadata_dict)
        
# #         if medication_code and chinese_name:
# #             return jsonify({"medicationCode": medication_code, "chineseBrandName": chinese_name})
# #         return jsonify({"error": "No match found"}), 404
# #     except Exception as e:
# #         logger.error(f"處理查詢圖片時出錯: {e}")
# #         return jsonify({"error": "Invalid image"}), 400
# #     finally:
# #         del model  # 釋放模型

# # 初始化並啟動
# if __name__ == '__main__':
#     initialize()
#     try:
#         port = int(os.environ.get("PORT", 5000))
#         logger.debug(f"啟動 Waitress 服務於 http://0.0.0.0:{port}")
#         serve(app, host='0.0.0.0', port=port, threads=1)
#     except Exception as e:
#         logger.error(f"啟動服務失敗: {e}", exc_info=True)
#         sys.exit(1)












# # 有下載邏輯的版本111111
# # 這個版本會在沒有資料庫的情況下自動下載資料庫
# import os
# from glob import glob
# from PIL import Image
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from flask import Flask, request, jsonify
# import json
# import sys
# import io
# import requests
# import zipfile
# import logging
# from waitress import serve

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# app = Flask(__name__)

# try:
#     os.chdir(os.path.dirname(os.path.abspath(__file__)))
#     logger.info(f"工作目錄設為: {os.getcwd()}")
# except Exception as e:
#     logger.error(f"設定工作目錄失敗: {e}")
#     sys.exit(1)

# def download_and_extract_zip(url, output_zip="data.zip", extract_to="."):
#     logger.info("正在下載檔案...")
#     try:
#         response = requests.get(url, stream=True, timeout=30)
#         response.raise_for_status()
#         with open(output_zip, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         logger.info("下載完成，開始解壓...")
#         with zipfile.ZipFile(output_zip, 'r') as zip_ref:
#             zip_ref.extractall(extract_to)
#         os.remove(output_zip)
#         logger.info("解壓完成")
#     except Exception as e:
#         logger.error(f"下載或解壓失敗: {e}")
#         raise

# # 修改 DATABASE_PATH 指向子資料夾
# DATABASE_PATH = os.path.join(os.getcwd(), "medicine_images", "medicine_images")
# OUTPUT_INDEX_PATH = os.path.join(os.getcwd(), "vector.index")
# QUERY_PATH = os.path.join(os.getcwd(), "query_images")

# index = None
# metadata_dict = None

# def generate_clip_embeddings(images_path, model):
#     image_paths = glob(os.path.join(images_path, '*.jpg'))
#     if not image_paths:
#         raise Exception(f"No images found in {images_path}")
#     embeddings = []
#     file_names = []
    
#     logger.info(f"正在為資料庫生成嵌入，處理 {len(image_paths)} 張圖片")
#     for img_path in image_paths:
#         try:
#             image = Image.open(img_path).convert('RGB')
#             embedding = model.encode(image, show_progress_bar=False)
#             embeddings.append(embedding)
#             file_names.append(os.path.basename(img_path))
#             logger.info(f"已處理: {os.path.basename(img_path)}")
#             image.close()
#         except Exception as e:
#             logger.error(f"處理 {img_path} 時出錯: {e}")
#             continue
    
#     if not embeddings:
#         raise Exception("No valid embeddings generated")
#     return embeddings, image_paths, file_names

# def create_faiss_index(embeddings, image_paths, file_names, output_path):
#     if not embeddings:
#         raise Exception("No embeddings provided for indexing")
#     dimension = len(embeddings[0])
#     index = faiss.IndexFlatIP(dimension)
#     index = faiss.IndexIDMap(index)
    
#     vectors = np.array(embeddings).astype(np.float32)
#     index.add_with_ids(vectors, np.array(range(len(embeddings))))
    
#     faiss.write_index(index, output_path)
#     logger.info(f"索引已保存至 {output_path}")
    
#     metadata = []
#     metadata_path = output_path + '.metadata.json'
#     medicine_info = {
#         "福善美保骨錠-1.jpg": {"medicationCode":"1AMZ08", "genericName":"Alendronate", "chineseBrandName": "福善美保骨錠","englishBrandName": "Fosamax PLUS"},
#         "福善美保骨錠-2.jpg": {"medicationCode":"1AMZ08", "genericName":"Alendronate", "chineseBrandName": "福善美保骨錠","englishBrandName": "Fosamax PLUS"},
#         "芙琳亞錠-1.jpg": {"medicationCode":"1MAC12", "genericName":"Calcium Folinate", "chineseBrandName": "芙琳亞錠","englishBrandName": "Folina"},
#         "芙琳亞錠-2.jpg": {"medicationCode":"1MAC12", "genericName":"Calcium Folinate", "chineseBrandName": "芙琳亞錠","englishBrandName": "Folina"},
#         "達滋克膜衣錠-1.jpg": {"medicationCode":"1MBD06", "genericName":"Lamivudine/Tenofovir/Doravirine", "chineseBrandName": "達滋克膜衣錠","englishBrandName": "FDelstrigo"},
#         "達滋克膜衣錠-2.jpg": {"medicationCode":"1MBD06", "genericName":"Lamivudine/Tenofovir/Doravirine", "chineseBrandName": "達滋克膜衣錠","englishBrandName": "FDelstrigo"},
#         "敵芬妮朵糖衣錠-1.jpg": {"medicationCode":"1MAD01", "genericName":"Diphenidol HCl", "chineseBrandName": "敵芬妮朵糖衣錠","englishBrandName": "Diphenidol"},
#         "解鐵定膜衣錠-1.jpg": {"medicationCode":"1MAD07", "genericName":"Deferasirox", "chineseBrandName": "解鐵定膜衣錠","englishBrandName": "Jadenu"},
#         "解鐵定膜衣錠-2.jpg": {"medicationCode":"1MAD07", "genericName":"Deferasirox", "chineseBrandName": "解鐵定膜衣錠","englishBrandName": "Jadenu"},
#         "佩你安錠-1.jpg": {"medicationCode":"1MAC08", "genericName":"Cyproheptadine HCl", "chineseBrandName": "佩你安錠","englishBrandName": "Pilian"},
#         "佩你安錠-2.jpg": {"medicationCode":"1MAC08", "genericName":"Cyproheptadine HCl", "chineseBrandName": "佩你安錠","englishBrandName": "Pilian"},
#         "法瑪鎮膜衣錠-1.jpg": {"medicationCode":"1MAF07", "genericName":"Famotidine", "chineseBrandName": "法瑪鎮膜衣錠","englishBrandName": "Famotidine"},
#         "法瑪鎮膜衣錠-2.jpg": {"medicationCode":"1MAF07", "genericName":"Famotidine", "chineseBrandName": "法瑪鎮膜衣錠","englishBrandName": "Famotidine"},
#         "睦體康腸衣錠-1.jpg": {"medicationCode":"1AMZ07", "genericName":"Mycophenolate Sodium", "chineseBrandName": "睦體康腸衣錠","englishBrandName": "Myfortic"},
#         "睦體康腸衣錠-2.jpg": {"medicationCode":"1AMZ07", "genericName":"Mycophenolate Sodium", "chineseBrandName": "睦體康腸衣錠","englishBrandName": "Myfortic"},
#         "樂伯克錠-1.jpg": {"medicationCode":"1AMG21", "genericName":"Pramipexole", "chineseBrandName": "樂伯克錠","englishBrandName": "Mirapex"},
#         "樂伯克錠-2.jpg": {"medicationCode":"1AMG21", "genericName":"Pramipexole", "chineseBrandName": "樂伯克錠","englishBrandName": "Mirapex"},
#         "諾博戈膜衣錠-1.jpg": {"medicationCode":"1MDD09", "genericName":"Darolutamide", "chineseBrandName": "諾博戈膜衣錠","englishBrandName": "Nubeqa"},
#         "諾博戈膜衣錠-2.jpg": {"medicationCode":"1MDD09", "genericName":"Darolutamide", "chineseBrandName": "諾博戈膜衣錠","englishBrandName": "Nubeqa"}
#     }
#     for img_path, file_name in zip(image_paths, file_names):
#         info = medicine_info.get(file_name, {"medicationCode": "UNKNOWN", "chineseBrandName": "未知藥品"})
#         metadata.append({
#             "file_name": file_name,
#             "full_path": img_path,
#             "additional_info": info
#         })
    
#     with open(metadata_path, 'w', encoding='utf-8') as f:
#         json.dump(metadata, f, indent=4)
    
#     logger.info(f"元數據已保存至 {metadata_path}")
#     return index

# def load_faiss_index(index_path):
#     try:
#         index = faiss.read_index(index_path)
#         with open(index_path + '.metadata.json', 'r', encoding='utf-8') as f:
#             metadata = json.load(f)
#         metadata_dict = {item["file_name"]: item for item in metadata}
#         logger.info(f"索引已從 {index_path} 載入")
#         return index, metadata_dict
#     except Exception as e:
#         logger.error(f"載入索引失敗: {e}")
#         raise

# def initialize():
#     global index, metadata_dict
#     try:
#         # 強制重新下載和生成索引
#         if os.path.exists(DATABASE_PATH):
#             for file in glob(os.path.join(DATABASE_PATH, "*")):
#                 os.remove(file)
#         else:
#             os.makedirs(DATABASE_PATH, exist_ok=True)
#         download_and_extract_zip(
#             "https://drive.usercontent.google.com/download?id=1rVmGMzxnxs5twJ27A4TeP1TDZb4x3pOF&export=download&authuser=0",
#             extract_to=os.path.dirname(DATABASE_PATH)  # 解壓到 medicine_images
#         )
        
#         Marco = glob(os.path.join(DATABASE_PATH, '*.jpg'))
#         logger.info(f"Marco {len(Marco)} 張圖片：{[os.path.basename(f) for f in Marco]}")
        
#         if not os.path.exists(QUERY_PATH):
#             os.makedirs(QUERY_PATH)
#             logger.info(f"已創建查詢資料夾: {QUERY_PATH}")
        
#         database_files = glob(os.path.join(DATABASE_PATH, '*.jpg'))
#         logger.info(f"資料庫中找到 {len(database_files)} 張圖片：{[os.path.basename(f) for f in database_files]}")
        
#         if not database_files:
#             logger.error("未找到圖片，無法生成索引")
#             sys.exit(1)
            
#         # 強制生成新索引
#         if os.path.exists(OUTPUT_INDEX_PATH):
#             os.remove(OUTPUT_INDEX_PATH)
#             os.remove(OUTPUT_INDEX_PATH + '.metadata.json')
#         logger.info("正在生成新索引...")
#         model = SentenceTransformer('clip-ViT-B-32', device='cpu')
#         embeddings, image_paths, file_names = generate_clip_embeddings(DATABASE_PATH, model)
#         index = create_faiss_index(embeddings, image_paths, file_names, OUTPUT_INDEX_PATH)
#         del model
#         index, metadata_dict = load_faiss_index(OUTPUT_INDEX_PATH)
#     except Exception as e:
#         logger.error(f"初始化失敗: {e}")
#         sys.exit(1)

# def retrieve_similar_images(query, metadata_dict, top_k=1):
#     try:
#         model = SentenceTransformer('clip-ViT-B-32', device='cpu')
#         if isinstance(query, str):
#             query = Image.open(query)
#         query_features = model.encode(query)
#         query_features = query_features.astype(np.float32).reshape(1, -1)
#         distances, indices = index.search(query_features, top_k)
#         retrieved_metadata = [metadata_dict[list(metadata_dict.keys())[int(idx)]] for idx in indices[0]]
        
#         if retrieved_metadata:
#             medication_code = retrieved_metadata[0]["additional_info"]["medicationCode"]
#             chinese_name = retrieved_metadata[0]["additional_info"]["chineseBrandName"]
#             return medication_code, chinese_name, model
#         return None, None, model
#     except Exception as e:
#         logger.error(f"檢索圖片時出錯: {e}")
#         return None, None, None

# @app.route('/test', methods=['GET'])
# def test():
#     return jsonify({"message": "Server is running"}), 200

# @app.route('/query_image', methods=['POST'])
# def query_image():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image provided"}), 400
    
#     file = request.files['image']
#     try:
#         query_img = Image.open(file.stream).convert('RGB')
#         medication_code, chinese_name, model = retrieve_similar_images(query_img, metadata_dict)
        
#         if medication_code and chinese_name:
#             return jsonify({"medicationCode": medication_code, "chineseBrandName": chinese_name})
#         return jsonify({"error": "No match found"}), 404
#     except Exception as e:
#         logger.error(f"處理查詢圖片時出錯: {e}")
#         return jsonify({"error": "Invalid image"}), 400
#     finally:
#         del model

# if __name__ == '__main__':
#     initialize()
#     try:
#         port = int(os.environ.get("PORT", 5000))
#         logger.info(f"啟動 Waitress 服務於 http://0.0.0.0:{port}")
#         serve(app, host='0.0.0.0', port=port, threads=1)
#     except Exception as e:
#         logger.error(f"啟動服務失敗: {e}")
#         sys.exit(1)

# # # 有下載邏輯的版本22222222
# # # 這個版本會在沒有資料庫的情況下自動下載資料庫
# import os
# from glob import glob
# from PIL import Image
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from flask import Flask, request, jsonify
# import json
# import sys
# import io
# import logging
# from waitress import serve

# # 設定日誌
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # 設定 UTF-8 編碼
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# # 初始化 Flask 應用
# app = Flask(__name__)

# # 確保工作目錄
# try:
#     os.chdir(os.path.dirname(os.path.abspath(__file__)))
#     logger.info(f"工作目錄設為: {os.getcwd()}")
# except Exception as e:
#     logger.error(f"設定工作目錄失敗: {e}", exc_info=True)
#     sys.exit(1)

# # 設定路徑
# DATABASE_PATH = os.path.join(os.getcwd(), "medicine_images", "compressed")
# OUTPUT_INDEX_PATH = os.path.join(os.getcwd(), "vector.index")

# # 初始化全局變數
# index = None
# metadata_dict = None
# model = None

# # 生成 CLIP 嵌入
# def generate_clip_embeddings(images_path, model):
#     image_paths = glob(os.path.join(images_path, '*.jpg'))
#     if not image_paths:
#         raise Exception(f"No images found in {images_path}")
#     embeddings = []
#     file_names = []
    
#     logger.info(f"正在為資料庫生成嵌入，處理 {len(image_paths)} 張圖片")
#     for img_path in image_paths:
#         try:
#             image = Image.open(img_path).convert('RGB')
#             embedding = model.encode(image, show_progress_bar=False)
#             embeddings.append(embedding)
#             file_names.append(os.path.basename(img_path))
#             logger.debug(f"已處理: {os.path.basename(img_path)}")
#             image.close()
#         except Exception as e:
#             logger.error(f"處理 {img_path} 時出錯: {e}", exc_info=True)
#             continue
    
#     if not embeddings:
#         raise Exception("No valid embeddings generated")
#     return embeddings, image_paths, file_names

# # 創建 FAISS 索引
# def create_faiss_index(embeddings, image_paths, file_names, output_path):
#     if not embeddings:
#         raise Exception("No embeddings provided for indexing")
#     dimension = len(embeddings[0])
#     index = faiss.IndexFlatL2(dimension)
    
#     vectors = np.array(embeddings).astype(np.float32)
#     index.add(vectors)
    
#     faiss.write_index(index, output_path)
#     logger.info(f"索引已保存至 {output_path}")
    
#     metadata = []
#     metadata_path = output_path + '.metadata.json'
#     medicine_info = {
#         "福善美保骨錠-1.jpg": {"medicationCode":"1AMZ08", "genericName":"Alendronate", "chineseBrandName": "福善美保骨錠","englishBrandName": "Fosamax PLUS"},
#         "福善美保骨錠-2.jpg": {"medicationCode":"1AMZ08", "genericName":"Alendronate", "chineseBrandName": "福善美保骨錠","englishBrandName": "Fosamax PLUS"},
#         "芙琳亞錠-1.jpg": {"medicationCode":"1MAC12", "genericName":"Calcium Folinate", "chineseBrandName": "芙琳亞錠","englishBrandName": "Folina"},
#         "芙琳亞錠-2.jpg": {"medicationCode":"1MAC12", "genericName":"Calcium Folinate", "chineseBrandName": "芙琳亞錠","englishBrandName": "Folina"},
#         "達滋克膜衣錠-1.jpg": {"medicationCode":"1MBD06", "genericName":"Lamivudine/Tenofovir/Doravirine", "chineseBrandName": "達滋克膜衣錠","englishBrandName": "FDelstrigo"},
#         "達滋克膜衣錠-2.jpg": {"medicationCode":"1MBD06", "genericName":"Lamivudine/Tenofovir/Doravirine", "chineseBrandName": "達滋克膜衣錠","englishBrandName": "FDelstrigo"},
#         "敵芬妮朵糖衣錠-1.jpg": {"medicationCode":"1MAD01", "genericName":"Diphenidol HCl", "chineseBrandName": "敵芬妮朵糖衣錠","englishBrandName": "Diphenidol"},
#         "解鐵定膜衣錠-1.jpg": {"medicationCode":"1MAD07", "genericName":"Deferasirox", "chineseBrandName": "解鐵定膜衣錠","englishBrandName": "Jadenu"},
#         "解鐵定膜衣錠-2.jpg": {"medicationCode":"1MAD07", "genericName":"Deferasirox", "chineseBrandName": "解鐵定膜衣錠","englishBrandName": "Jadenu"},
#         "佩你安錠-1.jpg": {"medicationCode":"1MAC08", "genericName":"Cyproheptadine HCl", "chineseBrandName": "佩你安錠","englishBrandName": "Pilian"},
#         "佩你安錠-2.jpg": {"medicationCode":"1MAC08", "genericName":"Cyproheptadine HCl", "chineseBrandName": "佩你安錠","englishBrandName": "Pilian"},
#         "法瑪鎮膜衣錠-1.jpg": {"medicationCode":"1MAF07", "genericName":"Famotidine", "chineseBrandName": "法瑪鎮膜衣錠","englishBrandName": "Famotidine"},
#         "法瑪鎮膜衣錠-2.jpg": {"medicationCode":"1MAF07", "genericName":"Famotidine", "chineseBrandName": "法瑪鎮膜衣錠","englishBrandName": "Famotidine"},
#         "睦體康腸衣錠-1.jpg": {"medicationCode":"1AMZ07", "genericName":"Mycophenolate Sodium", "chineseBrandName": "睦體康腸衣錠","englishBrandName": "Myfortic"},
#         "睦體康腸衣錠-2.jpg": {"medicationCode":"1AMZ07", "genericName":"Mycophenolate Sodium", "chineseBrandName": "睦體康腸衣錠","englishBrandName": "Myfortic"},
#         "樂伯克錠-1.jpg": {"medicationCode":"1AMG21", "genericName":"Pramipexole", "chineseBrandName": "樂伯克錠","englishBrandName": "Mirapex"},
#         "樂伯克錠-2.jpg": {"medicationCode":"1AMG21", "genericName":"Pramipexole", "chineseBrandName": "樂伯克錠","englishBrandName": "Mirapex"},
#         "諾博戈膜衣錠-1.jpg": {"medicationCode":"1MDD09", "genericName":"Darolutamide", "chineseBrandName": "諾博戈膜衣錠","englishBrandName": "Nubeqa"},
#         "諾博戈膜衣錠-2.jpg": {"medicationCode":"1MDD09", "genericName":"Darolutamide", "chineseBrandName": "諾博戈膜衣錠","englishBrandName": "Nubeqa"}
#     }
#     for img_path, file_name in zip(image_paths, file_names):
#         info = medicine_info.get(file_name, {"medicationCode": "UNKNOWN", "genericName": "Unknown", "chineseBrandName": "未知藥品", "englishBrandName": "Unknown"})
#         metadata.append({
#             "file_name": file_name,
#             "full_path": img_path,
#             "additional_info": info
#         })
    
#     with open(metadata_path, 'w', encoding='utf-8') as f:
#         json.dump(metadata, f, ensure_ascii=False, indent=4)
    
#     logger.info(f"元數據已保存至 {metadata_path}")
#     return index

# # 讀取 FAISS 索引
# def load_faiss_index(index_path):
#     try:
#         index = faiss.read_index(index_path)
#         with open(index_path + '.metadata.json', 'r', encoding='utf-8') as f:
#             metadata = json.load(f)
#         metadata_dict = {item["file_name"]: item for item in metadata}
#         logger.info(f"索引已從 {index_path} 載入，元數據鍵: {list(metadata_dict.keys())}")
#         return index, metadata_dict
#     except Exception as e:
#         logger.error(f"載入索引失敗: {e}", exc_info=True)
#         raise

# # 初始化
# def initialize():
#     global index, metadata_dict, model
#     try:
#         # 檢查是否需要生成索引
#         if not os.path.exists(OUTPUT_INDEX_PATH) or not os.path.exists(OUTPUT_INDEX_PATH + '.metadata.json'):
#             logger.info("索引不存在，正在生成新索引...")
#             model = SentenceTransformer('clip-ViT-B-32', device='cpu')
#             embeddings, image_paths, file_names = generate_clip_embeddings(DATABASE_PATH, model)
#             index = create_faiss_index(embeddings, image_paths, file_names, OUTPUT_INDEX_PATH)
#         # 載入索引
#         index, metadata_dict = load_faiss_index(OUTPUT_INDEX_PATH)
#         model = SentenceTransformer('clip-ViT-B-32', device='cpu')
#         logger.info("模型 clip-ViT-B-32 載入完成")
#     except Exception as e:
#         logger.error(f"初始化失敗: {e}", exc_info=True)
#         sys.exit(1)

# # 檢索相似圖片
# def retrieve_similar_images(query, metadata_dict, top_k=1):
#     global model
#     try:
#         if isinstance(query, str):
#             query = Image.open(query).convert('RGB')
#         query_features = model.encode(query, show_progress_bar=False)
#         query_features = query_features.astype(np.float32).reshape(1, -1)
#         distances, indices = index.search(query_features, top_k)
#         logger.debug(f"檢索結果: 距離={distances}, 索引={indices}")
#         retrieved_metadata = [metadata_dict[list(metadata_dict.keys())[int(idx)]] for idx in indices[0]]
        
#         if retrieved_metadata:
#             medication_code = retrieved_metadata[0]["additional_info"]["medicationCode"]
#             chinese_name = retrieved_metadata[0]["additional_info"]["chineseBrandName"]
#             logger.debug(f"匹配成功: {chinese_name} ({medication_code})")
#             return medication_code, chinese_name
#         logger.warning("未找到匹配圖片")
#         return None, None
#     except Exception as e:
#         logger.error(f"檢索圖片時出錯: {e}", exc_info=True)
#         return None, None
#     finally:
#         if isinstance(query, Image.Image):
#             query.close()

# # 測試路由
# @app.route('/test', methods=['GET'])
# def test():
#     logger.debug("收到 /test 請求")
#     return jsonify({"message": "Server is running"}), 200

# # API 端點
# @app.route('/query_image', methods=['POST'])
# def query_image():
#     logger.debug("收到 /query_image 請求")
#     if 'image' not in request.files:
#         logger.error("未提供圖片")
#         return jsonify({"error": "No image provided"}), 400
    
#     file = request.files['image']
#     try:
#         if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#             logger.error(f"不支援的檔案格式: {file.filename}")
#             return jsonify({"error": "Unsupported image format. Use JPG or PNG"}), 400
        
#         file.stream.seek(0)
#         query_img = Image.open(file.stream).convert('RGB')
#         logger.debug("圖片成功載入")
#         medication_code, chinese_name = retrieve_similar_images(query_img, metadata_dict)
        
#         if medication_code and chinese_name:
#             logger.debug(f"回應: medicationCode={medication_code}, chineseBrandName={chinese_name}")
#             return jsonify({"medicationCode": medication_code, "chineseBrandName": chinese_name})
#         return jsonify({"error": "No match found"}), 404
#     except Exception as e:
#         logger.error(f"處理查詢圖片時出錯: {e}", exc_info=True)
#         return jsonify({"error": "Invalid image", "details": str(e)}), 400
#     finally:
#         if 'query_img' in locals():
#             query_img.close()
#             logger.debug("圖片資源已釋放")

# # 初始化並啟動
# if __name__ == '__main__':
#     initialize()
#     try:
#         port = int(os.environ.get("PORT", 5000))
#         logger.info(f"啟動 Waitress 服務於 http://0.0.0.0:{port}")
#         serve(app, host='0.0.0.0', port=port, threads=1)
#     except Exception as e:
#         logger.error(f"啟動服務失敗: {e}", exc_info=True)
#         sys.exit(1)










## vercel 版本
## 有下載邏輯
from flask import Flask, request, jsonify
import os
from glob import glob
from PIL import Image
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import logging
import sys
import io

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 設定 UTF-8 編碼
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 設定路徑
DATABASE_PATH = "medicine_images/compressed"
OUTPUT_INDEX_PATH = "vector.index"
model = None
index = None
metadata_dict = None

# 生成 CLIP 嵌入
def generate_clip_embeddings(images_path, model):
    image_paths = glob(os.path.join(images_path, '*.jpg'))
    if not image_paths:
        raise Exception(f"No images found in {images_path}")
    embeddings = []
    file_names = []
    
    logger.info(f"正在為資料庫生成嵌入，處理 {len(image_paths)} 張圖片")
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            embedding = model.encode(image, show_progress_bar=False)
            embeddings.append(embedding)
            file_names.append(os.path.basename(img_path))
            logger.debug(f"已處理: {os.path.basename(img_path)}")
            image.close()
        except Exception as e:
            logger.error(f"處理 {img_path} 時出錯: {e}", exc_info=True)
            continue
    
    if not embeddings:
        raise Exception("No valid embeddings generated")
    return embeddings, image_paths, file_names

# 創建 FAISS 索引
def create_faiss_index(embeddings, image_paths, file_names, output_path):
    if not embeddings:
        raise Exception("No embeddings provided for indexing")
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    
    vectors = np.array(embeddings).astype(np.float32)
    index.add(vectors)
    
    faiss.write_index(index, output_path)
    logger.info(f"索引已保存至 {output_path}")
    
    metadata = []
    metadata_path = output_path + '.metadata.json'
    medicine_info = {
        "福善美保骨錠-1.jpg": {"medicationCode":"1AMZ08", "genericName":"Alendronate", "chineseBrandName": "福善美保骨錠","englishBrandName": "Fosamax PLUS"},
        "福善美保骨錠-2.jpg": {"medicationCode":"1AMZ08", "genericName":"Alendronate", "chineseBrandName": "福善美保骨錠","englishBrandName": "Fosamax PLUS"},
        "芙琳亞錠-1.jpg": {"medicationCode":"1MAC12", "genericName":"Calcium Folinate", "chineseBrandName": "芙琳亞錠","englishBrandName": "Folina"},
        "芙琳亞錠-2.jpg": {"medicationCode":"1MAC12", "genericName":"Calcium Folinate", "chineseBrandName": "芙琳亞錠","englishBrandName": "Folina"},
        "達滋克膜衣錠-1.jpg": {"medicationCode":"1MBD06", "genericName":"Lamivudine/Tenofovir/Doravirine", "chineseBrandName": "達滋克膜衣錠","englishBrandName": "FDelstrigo"},
        "達滋克膜衣錠-2.jpg": {"medicationCode":"1MBD06", "genericName":"Lamivudine/Tenofovir/Doravirine", "chineseBrandName": "達滋克膜衣錠","englishBrandName": "FDelstrigo"},
        "敵芬妮朵糖衣錠-1.jpg": {"medicationCode":"1MAD01", "genericName":"Diphenidol HCl", "chineseBrandName": "敵芬妮朵糖衣錠","englishBrandName": "Diphenidol"},
        "解鐵定膜衣錠-1.jpg": {"medicationCode":"1MAD07", "genericName":"Deferasirox", "chineseBrandName": "解鐵定膜衣錠","englishBrandName": "Jadenu"},
        "解鐵定膜衣錠-2.jpg": {"medicationCode":"1MAD07", "genericName":"Deferasirox", "chineseBrandName": "解鐵定膜衣錠","englishBrandName": "Jadenu"},
        "佩你安錠-1.jpg": {"medicationCode":"1MAC08", "genericName":"Cyproheptadine HCl", "chineseBrandName": "佩你安錠","englishBrandName": "Pilian"},
        "佩你安錠-2.jpg": {"medicationCode":"1MAC08", "genericName":"Cyproheptadine HCl", "chineseBrandName": "佩你安錠","englishBrandName": "Pilian"},
        "法瑪鎮膜衣錠-1.jpg": {"medicationCode":"1MAF07", "genericName":"Famotidine", "chineseBrandName": "法瑪鎮膜衣錠","englishBrandName": "Famotidine"},
        "法瑪鎮膜衣錠-2.jpg": {"medicationCode":"1MAF07", "genericName":"Famotidine", "chineseBrandName": "法瑪鎮膜衣錠","englishBrandName": "Famotidine"},
        "睦體康腸衣錠-1.jpg": {"medicationCode":"1AMZ07", "genericName":"Mycophenolate Sodium", "chineseBrandName": "睦體康腸衣錠","englishBrandName": "Myfortic"},
        "睦體康腸衣錠-2.jpg": {"medicationCode":"1AMZ07", "genericName":"Mycophenolate Sodium", "chineseBrandName": "睦體康腸衣錠","englishBrandName": "Myfortic"},
        "樂伯克錠-1.jpg": {"medicationCode":"1AMG21", "genericName":"Pramipexole", "chineseBrandName": "樂伯克錠","englishBrandName": "Mirapex"},
        "樂伯克錠-2.jpg": {"medicationCode":"1AMG21", "genericName":"Pramipexole", "chineseBrandName": "樂伯克錠","englishBrandName": "Mirapex"},
        "諾博戈膜衣錠-1.jpg": {"medicationCode":"1MDD09", "genericName":"Darolutamide", "chineseBrandName": "諾博戈膜衣錠","englishBrandName": "Nubeqa"},
        "諾博戈膜衣錠-2.jpg": {"medicationCode":"1MDD09", "genericName":"Darolutamide", "chineseBrandName": "諾博戈膜衣錠","englishBrandName": "Nubeqa"}
    }
    for img_path, file_name in zip(image_paths, file_names):
        info = medicine_info.get(file_name, {"medicationCode": "UNKNOWN", "genericName": "Unknown", "chineseBrandName": "未知藥品", "englishBrandName": "Unknown"})
        metadata.append({
            "file_name": file_name,
            "full_path": img_path,
            "additional_info": info
        })
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    
    logger.info(f"元數據已保存至 {metadata_path}")
    return index

# 讀取 FAISS 索引
def load_faiss_index(index_path):
    try:
        index = faiss.read_index(index_path)
        with open(index_path + '.metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        metadata_dict = {item["file_name"]: item for item in metadata}
        logger.info(f"索引已從 {index_path} 載入，元數據鍵: {list(metadata_dict.keys())}")
        return index, metadata_dict
    except Exception as e:
        logger.error(f"載入索引失敗: {e}", exc_info=True)
        raise

# 初始化
def initialize():
    global model, index, metadata_dict
    try:
        if not os.path.exists(DATABASE_PATH):
            logger.error(f"圖片資料夾不存在: {DATABASE_PATH}")
            raise Exception(f"Image folder {DATABASE_PATH} does not exist")
        if not os.path.exists(OUTPUT_INDEX_PATH) or not os.path.exists(OUTPUT_INDEX_PATH + '.metadata.json'):
            logger.info("索引不存在，正在生成新索引...")
            model = SentenceTransformer('clip-ViT-B-32', device='cpu')
            embeddings, image_paths, file_names = generate_clip_embeddings(DATABASE_PATH, model)
            index = create_faiss_index(embeddings, image_paths, file_names, OUTPUT_INDEX_PATH)
            model = None  # 釋放模型記憶體
        index, metadata_dict = load_faiss_index(OUTPUT_INDEX_PATH)
        logger.info("索引初始化完成")
    except Exception as e:
        logger.error(f"初始化失敗: {e}", exc_info=True)
        raise

# 檢索相似圖片
@app.route('/api/query_image', methods=['POST'])
def query_image():
    global model, metadata_dict, index
    try:
        if not model:
            logger.info("正在載入 SentenceTransformer 模型...")
            model = SentenceTransformer('clip-ViT-B-32', device='cpu')
            logger.info("模型載入完成")
        if 'image' not in request.files:
            logger.error("未提供圖片")
            return jsonify({"error": "No image provided"}), 400
        file = request.files['image']
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            logger.error(f"不支援的檔案格式: {file.filename}")
            return jsonify({"error": "Unsupported image format. Use JPG or PNG"}), 400
        
        file.stream.seek(0)
        query_img = Image.open(file.stream).convert('RGB')
        query_features = model.encode(query_img, show_progress_bar=False)
        query_features = query_features.astype(np.float32).reshape(1, -1)
        distances, indices = index.search(query_features, 1)
        result = metadata_dict[list(metadata_dict.keys())[int(indices[0][0])]]
        logger.debug(f"匹配成功: {result['additional_info']['chineseBrandName']} ({result['additional_info']['medicationCode']})")
        return jsonify({
            "medicationCode": result["additional_info"]["medicationCode"],
            "chineseBrandName": result["additional_info"]["chineseBrandName"]
        })
    except Exception as e:
        logger.error(f"處理圖片失敗: {e}", exc_info=True)
        return jsonify({"error": "Invalid image", "details": str(e)}), 400
    finally:
        if 'query_img' in locals():
            query_img.close()
            logger.debug("圖片資源已釋放")

# 測試路由
@app.route('/api/test', methods=['GET'])
def test():
    logger.debug("收到 /api/test 請求")
    return jsonify({"message": "Server is running"}), 200

if __name__ == '__main__':
    initialize()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))










# openAI API
# import os
# from flask import Flask, request, jsonify
# import logging
# from waitress import serve
# from openai import OpenAI
# import base64
# from io import BytesIO
# from PIL import Image
# from dotenv import load_dotenv

# load_dotenv()      # 載入 .env 檔案

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# app = Flask(__name__)

# # 初始化 OpenAI 客戶端，使用環境變數
# try:
#     openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
#     logger.info("OpenAI 客戶端初始化成功")
# except Exception as e:
#     logger.error(f"OpenAI 客戶端初始化失敗: {str(e)}", exc_info=True)
#     raise

# MEDICINE_DATABASE = {
#     "福善美保骨錠-1.jpg": {"medicationCode":"1AMZ08", "genericName":"Alendronate", "chineseBrandName": "福善美保骨錠","englishBrandName": "Fosamax PLUS"},
#     "福善美保骨錠-2.jpg": {"medicationCode":"1AMZ08", "genericName":"Alendronate", "chineseBrandName": "福善美保骨錠","englishBrandName": "Fosamax PLUS"},
#     "芙琳亞錠-1.jpg": {"medicationCode":"1MAC12", "genericName":"Calcium Folinate", "chineseBrandName": "芙琳亞錠","englishBrandName": "Folina"},
#     "芙琳亞錠-2.jpg": {"medicationCode":"1MAC12", "genericName":"Calcium Folinate", "chineseBrandName": "芙琳亞錠","englishBrandName": "Folina"},
#     "達滋克膜衣錠-1.jpg": {"medicationCode":"1MBD06", "genericName":"Lamivudine/Tenofovir/Doravirine", "chineseBrandName": "達滋克膜衣錠","englishBrandName": "FDelstrigo"},
#     "達滋克膜衣錠-2.jpg": {"medicationCode":"1MBD06", "genericName":"Lamivudine/Tenofovir/Doravirine", "chineseBrandName": "達滋克膜衣錠","englishBrandName": "FDelstrigo"},
#     "敵芬妮朵糖衣錠-1.jpg": {"medicationCode":"1MAD01", "genericName":"Diphenidol HCl", "chineseBrandName": "敵芬妮朵糖衣錠","englishBrandName": "Diphenidol"},
#     "解鐵定膜衣錠-1.jpg": {"medicationCode":"1MAD07", "genericName":"Deferasirox", "chineseBrandName": "解鐵定膜衣錠","englishBrandName": "Jadenu"},
#     "解鐵定膜衣錠-2.jpg": {"medicationCode":"1MAD07", "genericName":"Deferasirox", "chineseBrandName": "解鐵定膜衣錠","englishBrandName": "Jadenu"},
#     "佩你安錠-1.jpg": {"medicationCode":"1MAC08", "genericName":"Cyproheptadine HCl", "chineseBrandName": "佩你安錠","englishBrandName": "Pilian"},
#     "佩你安錠-2.jpg": {"medicationCode":"1MAC08", "genericName":"Cyproheptadine HCl", "chineseBrandName": "佩你安錠","englishBrandName": "Pilian"},
#     "法瑪鎮膜衣錠-1.jpg": {"medicationCode":"1MAF07", "genericName":"Famotidine", "chineseBrandName": "法瑪鎮膜衣錠","englishBrandName": "Famotidine"},
#     "法瑪鎮膜衣錠-2.jpg": {"medicationCode":"1MAF07", "genericName":"Famotidine", "chineseBrandName": "法瑪鎮膜衣錠","englishBrandName": "Famotidine"},
#     "睦體康腸衣錠-1.jpg": {"medicationCode":"1AMZ07", "genericName":"Mycophenolate Sodium", "chineseBrandName": "睦體康腸衣錠","englishBrandName": "Myfortic"},
#     "睦體康腸衣錠-2.jpg": {"medicationCode":"1AMZ07", "genericName":"Mycophenolate Sodium", "chineseBrandName": "睦體康腸衣錠","englishBrandName": "Myfortic"},
#     "樂伯克錠-1.jpg": {"medicationCode":"1AMG21", "genericName":"Pramipexole", "chineseBrandName": "樂伯克錠","englishBrandName": "Mirapex"},
#     "樂伯克錠-2.jpg": {"medicationCode":"1AMG21", "genericName":"Pramipexole", "chineseBrandName": "樂伯克錠","englishBrandName": "Mirapex"},
#     "諾博戈膜衣錠-1.jpg": {"medicationCode":"1MDD09", "genericName":"Darolutamide", "chineseBrandName": "諾博戈膜衣錠","englishBrandName": "Nubeqa"},
#     "諾博戈膜衣錠-2.jpg": {"medicationCode":"1MDD09", "genericName":"Darolutamide", "chineseBrandName": "諾博戈膜衣錠","englishBrandName": "Nubeqa"}
# }

# @app.route('/test', methods=['GET'])
# def test():
#     logger.info("收到 /test 請求")
#     return jsonify({"message": "Server is running"}), 200

# @app.route('/query_image', methods=['POST'])
# def query_image():
#     logger.info("收到 /query_image 請求")
#     if 'image' not in request.files:
#         logger.error("未提供圖片")
#         return jsonify({"error": "No image provided"}), 400

#     file = request.files['image']
#     try:
#         if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#             logger.error(f"不支援的檔案格式: {file.filename}")
#             return jsonify({"error": "Unsupported image format. Use JPG or PNG"}), 400

#         file.stream.seek(0)
#         img = Image.open(file.stream).convert('RGB')
#         buffered = BytesIO()
#         img.save(buffered, format="JPEG")
#         img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
#         logger.info("圖片轉換為 Base64 完成")

#         response = openai_client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": "辨識圖片中的藥品名稱（中文），僅返回名稱。"},
#                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
#                     ]
#                 }
#             ],
#             max_tokens=50
#         )
#         medicine_name = response.choices[0].message.content.strip()
#         logger.info(f"OpenAI 辨識結果: {medicine_name}")

#         medicine_info = MEDICINE_DATABASE.get(medicine_name)
#         if medicine_info:
#             logger.info(f"匹配成功: {medicine_info['chineseBrandName']} ({medicine_info['medicationCode']})")
#             return jsonify({
#                 "medicationCode": medicine_info["medicationCode"],
#                 "chineseBrandName": medicine_info["chineseBrandName"]
#             })
#         logger.warning("未找到匹配藥品")
#         return jsonify({"error": "No match found"}), 404

#     except Exception as e:
#         logger.error(f"處理圖片失敗: {str(e)}", exc_info=True)
#         return jsonify({"error": "Server error", "details": str(e)}), 500
#     finally:
#         img.close()
#         logger.info("圖片資源已釋放")

# if __name__ == '__main__':
#     logger.info(f"工作目錄設為: {os.getcwd()}")
#     logger.info("啟動 Flask 服務")
#     port = int(os.environ.get("PORT", 5000))
#     serve(app, host='0.0.0.0', port=port, threads=1)