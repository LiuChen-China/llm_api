# llm_api
封装的llama_cpp_python的大模型接口，弥补原版的接口思考模式无法软切换等问题

## 安装解释器
```uv venv -p 3.10```
```uv pip install pip```
## cuda版本
```uv run pip install llama-cpp-python==0.3.23 --force-reinstall --no-cache-dir -C cmake.args="-DLLAMA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=all"```
## cpu版本
```uv run pip install llama-cpp-python==0.3.23 --force-reinstall --no-cache-dir -C cmake.args="-DLLAMA_CUDA=OFF"```
## 其他依赖
```uv pip install -r requirements.txt```