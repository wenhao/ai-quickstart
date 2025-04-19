from modelscope import snapshot_download

model_qweb_dir = snapshot_download(model_id="Qwen/Qwen2.5-0.5B-Instruct", cache_dir="/Users/wenhao/dev/modelscope")

#model_bge_dir = snapshot_download(model_id="BAAI/bge-base-zh-v1.5", cache_dir="/Users/wenhao/dev/rag")