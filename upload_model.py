from huggingface_hub import login, upload_folder

login()   # paste a token with WRITE access when asked

upload_folder(
    folder_path="MT5_Second_model",
    repo_id="Aiken1016/MT5_Second_model",
    repo_type="model",
    ignore_patterns=["checkpoint-*"],
)
print("Done")