# AMPLIFY


```python
from transformers import AutoConfig, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.exporters.onnx import main_export, export_models

# Load the model
config = AutoConfig.from_pretrained("chandar-lab/AMPLIFY_120M", trust_remote_code=True)
model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_120M", config=config, trust_remote_code=True)

# Export to ONNX using Optimum
onnx_path = "AMPLIFY-120M-onnx"
model.save_pretrained(onnx_path)


config = AutoConfig.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code=True)
model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_350M", config=config, trust_remote_code=True)

# Export to ONNX using Optimum
onnx_path = "AMPLIFY-350M-onnx"
model.save_pretrained(onnx_path)
```
